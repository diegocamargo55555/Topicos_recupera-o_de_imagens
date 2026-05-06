import cv2, torch, os, numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

"""
#!/bin/bash
curl -L -o ~/Downloads/tobacco-800-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/sprytte/tobacco-800-dataset
"""

dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {dispositivo}")

modelo = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
modelo.fc = torch.nn.Identity()  
modelo.to(dispositivo).eval()

pre_processamento = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extrair_caracteristicas(recortes, tamanho_lote=32):
    if not recortes: return []
    todas_caracteristicas = []
    for i in range(0, len(recortes), tamanho_lote):
        lote = recortes[i:i+tamanho_lote]
        tensores = torch.stack([pre_processamento(Image.fromarray(c)) for c in lote]).to(dispositivo)
        with torch.no_grad():
            todas_caracteristicas.append(modelo(tensores).cpu().numpy())
    return np.vstack(todas_caracteristicas)

def obter_regioes(caminho_imagem):
    #Gera regiões candidatas usando dilatação e componentes conectados
    img = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
    if img is None: return []
    
    dilatada = cv2.dilate(img, np.ones((5, 5), np.uint8), iterations=2)
    _, _, estatisticas, _ = cv2.connectedComponentsWithStats(dilatada)
    alt_orig, larg_orig = img.shape
    
    regioes = []
    for x, y, l, a, area in estatisticas[1:]: # Ignora o fundo (background)
        if 1000 < area < 500000:
            recorte = cv2.cvtColor(img[y:y+a, x:x+l], cv2.COLOR_GRAY2RGB)
            regioes.append({
                'caminho': caminho_imagem, 
                'caixa': (x, y, l, a), 
                'normalizado': (x/larg_orig, y/alt_orig, l/larg_orig, a/alt_orig), 
                'recorte': recorte
            })
    return regioes

def calcular_pontuacao(caract_query, norm_query, caract_bd, norm_bd, alfa=0.7):
    
    # Similaridade de Cosseno (Visual)
    sim_visual = np.dot(caract_query, caract_bd) / (np.linalg.norm(caract_query) * np.linalg.norm(caract_bd))
    
    # IoU (Espacial)
    qx, qy, ql, qa = norm_query
    bx, by, bl, ba = norm_bd
    inter_x = max(0, min(qx+ql, bx+bl) - max(qx, bx))
    inter_y = max(0, min(qy+qa, by+ba) - max(qy, by))
    area_inter = inter_x * inter_y
    area_uniao = (ql*qa) + (bl*ba) - area_inter
    sim_espacial = area_inter / area_uniao if area_uniao > 0 else 0
    
    pontuacao_final = (alfa * sim_visual) + ((1 - alfa) * sim_espacial)
    return pontuacao_final, sim_visual, sim_espacial

def selecionar_regiao_interativo(caminho_imagem):
    img = cv2.imread(caminho_imagem)
    alt, larg = img.shape[:2]
    escala = 800 / max(alt, larg) if max(alt, larg) > 800 else 1.0
    imagem_exibicao = cv2.resize(img, (int(larg*escala), int(alt*escala)))
    
    print(f"\n[INTERATIVO] Selecione a região em {os.path.basename(caminho_imagem)}")
    
    roi = cv2.selectROI("Selecione a Regiao da Query", imagem_exibicao, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Selecione a Regiao da Query")
    
    if roi == (0,0,0,0): return None
    x, y, l_roi, a_roi = [int(v/escala) for v in roi]
    
    return {
        'recorte': cv2.cvtColor(img[y:y+a_roi, x:x+l_roi], cv2.COLOR_BGR2RGB), 
        'normalizado': (x/larg, y/alt, l_roi/larg, a_roi/alt), 
        'caixa': (x, y, l_roi, a_roi), 
        'imagem_completa': img
    }

if __name__ == "__main__":
    DIR_DADOS, DIR_RESULTADOS = "tobacco800", "resultados"
    
    if not os.path.exists(DIR_RESULTADOS):
        os.makedirs(DIR_RESULTADOS)
        print(f"Diretório criado: {DIR_RESULTADOS}")

    imagens = [os.path.join(DIR_DADOS, f) for f in sorted(os.listdir(DIR_DADOS)) if f.endswith(".png")]
    imagens_bd, documentos_consulta = imagens[:30], imagens[30:35]

    print(f"Indexando {len(imagens_bd)} imagens da base de dados...")
    metadados_bd = []
    for caminho in imagens_bd: metadados_bd.extend(obter_regioes(caminho))
    caracteristicas_bd = extrair_caracteristicas([r['recorte'] for r in metadados_bd])
    print(f"Concluído. Foram indexadas {len(caracteristicas_bd)} regiões candidatas.")

    print(f"\nIniciando seleção interativa para {len(documentos_consulta)} imagens de consulta...")
    consultas = []
    for caminho in documentos_consulta:
        selecao = selecionar_regiao_interativo(caminho)
        if selecao:
            vetor_query = extrair_caracteristicas([selecao['recorte']])[0]
            consultas.append({**selecao, 'caminho': caminho, 'vetor': vetor_query})
        else:
            print(f"Seleção cancelada para {os.path.basename(caminho)}. Pulando.")

    if not consultas:
        print("\nNenhuma consulta selecionada. Encerrando.")
        exit()

    print(f"\nRealizando busca para {len(consultas)} consultas selecionadas:")
    for i, q in enumerate(consultas):
        id_consulta, nome_arquivo = i + 1, os.path.basename(q['caminho'])
        print(f"\nPROCESSANDO CONSULTA {id_consulta}: {nome_arquivo}")
        
        pasta_consulta = os.path.join(DIR_RESULTADOS, f"query_{id_consulta}")
        os.makedirs(pasta_consulta, exist_ok=True)
        
        # Salva a imagem da Query (Consulta)
        qx, qy, ql, qa = q['caixa']
        img_query_salvar = q['imagem_completa'].copy()
        cv2.rectangle(img_query_salvar, (qx, qy), (qx+ql, qy+qa), (0,0,255), 3)
        cv2.imwrite(os.path.join(pasta_consulta, "00_consulta.png"), img_query_salvar)

        # Calcula similaridade e ordena resultados
        todos_resultados = sorted([{**r, 'pontos': calcular_pontuacao(q['vetor'], q['normalizado'], f, r['normalizado'])} 
                                  for f, r in zip(caracteristicas_bd, metadados_bd)], key=lambda x: x['pontos'][0], reverse=True)

        # Filtra para que cada resultado venha de uma imagem diferente
        resultados_unicos = []
        imagens_usadas = set()
        for res in todos_resultados:
            if res['caminho'] not in imagens_usadas:
                resultados_unicos.append(res)
                imagens_usadas.add(res['caminho'])
            if len(resultados_unicos) == 5:
                break

        with open(os.path.join(pasta_consulta, "resultados.txt"), "w", encoding="utf-8") as f_saida:
            f_saida.write(f"Resultados para Consulta Interativa {id_consulta}: {nome_arquivo}\n")
            f_saida.write("-" * 60 + "\n")

            for j, res in enumerate(resultados_unicos):
                score, sim_v, sim_e = res['pontos']
                txt_score = f"Score: {score:.3f} (Visual: {sim_v:.3f}, Espacial: {sim_e:.3f})"
                linha = f"  [{j+1}] {os.path.basename(res['caminho'])} | {txt_score}"
                print(linha)
                f_saida.write(linha + "\n")

                # Salva imagem do resultado 
                img_res = cv2.imread(res['caminho'])
                rx, ry, rl, ra = res['caixa']
                cv2.rectangle(img_res, (rx, ry), (rx+rl, ry+ra), (0,255,0), 3)
                cv2.imwrite(os.path.join(pasta_consulta, f"resultado_{j+1}.png"), img_res)

    print(f"\nResultados salvos na pasta '{DIR_RESULTADOS}'.")
    print("Execução do sistema CBIR concluída com sucesso.")
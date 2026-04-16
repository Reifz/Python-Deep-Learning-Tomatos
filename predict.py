import tensorflow as tf
import numpy as np
from PIL import Image
import os

IMG_SIZE = 224

# Tenta carregar o melhor modelo treinado recentemente, senao usa o v2_d1v
model_path = "best_tomato_model.h5"
if not os.path.exists(model_path):
    model_path = "tomato_model_v2_d1v.h5"

print(f" usando modelo: {model_path}")

# carregar modelo
model = tf.keras.models.load_model(model_path)

# carregar classes
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Erro: Arquivo {image_path} nao encontrado.")
        return

    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    # Importante: o modelo agora tem a camada de Rescaling interna, 
    # entao passamos a imagem no intervalo [0, 255]
    img_array = np.array(img).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]

    # combinar classe + probabilidade
    resultados = list(zip(class_names, predictions))

    # ordenar do maior pro menor
    resultados.sort(key=lambda x: x[1], reverse=True)

    print(f"\n📊 Probabilidades para: {image_path}\n")

    for i, (classe, prob) in enumerate(resultados[:5]): # mostra top 5
        print(f"{i+1}º - {classe}: {prob*100:.2f}%")

    # melhor resultado
    melhor_classe, melhor_prob = resultados[0]

    print("\n✅ Resultado final:")
    print(f"{melhor_classe} ({melhor_prob*100:.2f}%)")

# Testes com as imagens na raiz
# imagens_teste = ["teste.jpg","teste3.jpg","teste4.jpg","teste_s.jpg", "teste_bac.jpg", "teste.jpg"]
imagens_teste = [
    "tom_d1.png",
    "tom_d2.png",
    "tom_d3.png",
    "tom_f.png",
    "tom_f2.png",
    "tom_t1.png",
    "tom_t2.png",
    "tom_t3.png",
    "tom_t4.png",
    "tom_t5.jpg",
    "tom_t6.jpg",
    "tom_t7.jpg",
    "tom_t8.jpg",
    "tom_t9.jpg",
    "tom_t11.PNG",
    "tom_t12.JPG",
    "tom_t13.JPG"
]
for img in imagens_teste:
    predict_image(img)
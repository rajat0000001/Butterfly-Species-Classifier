from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing import image
from PIL import Image
import json
with open("class_indices.json", "r") as f:
    cs=json.load(f)

model=load_model("Butterfly_Species.h5")
st.title("Butterfly Classifier")
upf=st.file_uploader("Upload a buttrfly image", type=["jpg","png","jpeg"])

if upf:
    im=Image.open(upf).convert("RGB")
    st.image(im,caption="Uploaded Image", use_column_width=True)
    img=im.resize((224,224))
    imga=image.img_to_array(img)/255.0
    imga=np.expand_dims(imga,axis=0)

    prd=model.predict(imga)[0]

    prdc=np.argmax(prd)
    

    lb=dict((v,k) for k,v in cs.items())
    prl=lb[prdc]
    con=prd[prdc]
    
    st.success(lb[prdc])

    
    st.subheader(con)
    st.subheader("Other likely species:")

    for i in prd.argsort()[-3:][::1]:
        st.write(f"{lb[i]}:{prd[i]*100:.2f}")
import streamlit as st
import pickle
import numpy as np

pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title('Predict-a-Price: Your Laptop Price Predictor:computer:')
st.markdown('Welcome to Predict-a-Price, your ultimate destination for anticipating laptop prices. Our advanced algorithm analyzes market trends, historical data, and upcoming technological advancements to provide you with accurate price predictions for a wide range of laptops. Whether you are a savvy shopper planning your next purchase or a tech enthusiast wanting to stay ahead of the market, Predict-a-Price equips you with the insights needed to make informed decisions. Stay ahead of the curve with our reliable and user-friendly platform. Explore, Predict, and Save with Predict-a-Price!')


st.markdown('Mark your desired brand and laptop configurations below.')

company = st.selectbox('Select preferred Brand', df['Company'].unique())
type = st.selectbox('Select preferred Type', df['TypeName'].unique())

ram = st.selectbox('Select preferred RAM(in GB)',[2,4,6,8,12,16,24,32,64])
weight = st.number_input('Mention preferred Weight(in KG)', min_value=0.05)
touchscreen = st.selectbox('Want to have Touchscreen?',['Yes','No'])
ips = st.selectbox('Want to have IPS Display?',['Yes','No'])
screen_size = st.number_input('Mention preferred Screen Size', min_value=0.2)
resolution = st.selectbox('Select preferred Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
cpu = st.selectbox('Select preferred Processor',df['Cpu brand'].unique())
hdd = st.selectbox('Select preferred HDD(in GB)',[0,128,256,512,1024,2048])
ssd = st.selectbox('Select preferred SSD(in GB)',[0,8,128,256,512,1024])
gpu = st.selectbox('Select preferred Graphics Card',df['Gpu brand'].unique())
os = st.selectbox('Select preferred Operating System',df['os'].unique())

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0


    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query = query.reshape(1,12)
    st.markdown("The predicted price with this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))



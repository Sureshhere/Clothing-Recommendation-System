import pandas as pd
import streamlit as st
import pickle
from Recommender2 import get_combined_recommendations, finalImageData

products_dict = pickle.load(open('productAndImagesDict.pkl', 'rb'))
articleType_dict = pickle.load(open('articleTypeDict.pkl', 'rb'))
baseColour_dict = pickle.load(open('baseColourDict.pkl','rb'))


products = pd.DataFrame(products_dict)
articleTypes = pd.DataFrame(articleType_dict)
baseColours = pd.DataFrame(baseColour_dict)


st.set_page_config(
    page_title="Personalised Shopping Compananion",
    page_icon=":sunglasses:",
    layout="wide",
)

st.title("Clothing Recommender System")


column1, column2, column3 = st.columns([3,1,1])
with column1:
    productName = st.selectbox(
       "Product",
       products['productName'].values,
       index=None,
       placeholder="Search for item",
    )
with column2:
    articleTypeName = st.selectbox(
       "Category",
       articleTypes['articleType'].values,
       index=None,
       placeholder="Search by Category",
    )
with column3:
    baseColourName = st.selectbox(
       "Colour",
       baseColours['baseColour'].values,
       index=None,
       placeholder="Search for colour",
    )


try:
    if st.button('Get Recommendations'):

        recommendations = get_combined_recommendations(productName, articleTypeName, baseColourName)

        # Display recommendations
        st.subheader('Recommended Products:')
        for product in recommendations:
            product_info = finalImageData.loc[finalImageData['productName'] == product]
            if not product_info.empty:
                col1, col2 = st.columns([1, 3])  # Create two columns
                with col1:
                    image_url = product_info['link'].values[0]
                    st.image(image_url, caption='', use_column_width=False, width=150,
                             output_format='JPEG')  # Display image in the second column
                    st.markdown(
                        """
                        <style>
                        .css-1sjlbgp {
                            border: 2px solid red;  /* Add border style here */
                            padding: 10px;  /* Add padding for spacing */
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )
                    with col2:
                        st.markdown(product)

except (Exception):
    st.write("Please select any product")


#=============================================
# --------- CUSTOMIZATION -------------------
#=============================================

# hiding streamlit footer text
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# reducing header padding
reduce_header_height_style = """
    <style>
        div.block-container {padding-top:1rem;}
    </style>
"""
st.markdown(reduce_header_height_style, unsafe_allow_html=True)

hide_header="""
<style>
    header {
    visibility:hidden;
}
</style>

"""
st.markdown(hide_header,unsafe_allow_html=True)

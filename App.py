import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shapash.explainer.smart_explainer import SmartExplainer
import streamlit.components.v1 as components
import requests
from pathlib import Path

max_width_str = f"max-width: 1500px;"
st.set_page_config(
     page_title="Ex-stream-ly Cool App",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "# This is a header. This is an *extremely* cool app!"
     }
 )
st.markdown(
	f"""
		<style>
			.reportview-container .main .block-container {{{max_width_str}}}
		</style>
	""",
	unsafe_allow_html=True
)

############################## FONCTIONS ######################################################

pkl_path = Path(__file__).parents[1]
pickle_in = open('lgb_model_auc.pkl', 'rb')
clf = pickle.load(pickle_in)


def get_carac(chk_id:int):
    Genre = InfoClient_test[InfoClient_test["SK_ID_CURR"]==chk_id]['Sexe'].iloc[0]
    Statut_marital = InfoClient_test[InfoClient_test["SK_ID_CURR"]==chk_id]['Statut familial'].iloc[0]
    Enfant = InfoClient_test[InfoClient_test["SK_ID_CURR"]==chk_id]['CNT_CHILDREN'].iloc[0]
    Salaires = InfoClient_test[InfoClient_test["SK_ID_CURR"]==chk_id]['AMT_INCOME_TOTAL'].iloc[0]
    Montant_du_crédit = InfoClient_test[InfoClient_test["SK_ID_CURR"]==chk_id]['AMT_CREDIT'].iloc[0]
    Type_credit = InfoClient_test[InfoClient_test["SK_ID_CURR"]==chk_id]["Type du nom du contrat"].iloc[0]
    Historique_du_crédit = InfoClient_test[InfoClient_test["SK_ID_CURR"]==chk_id]['number_prev_loan_extern'].iloc[0]
    Jours_travaillés = InfoClient_test[InfoClient_test["SK_ID_CURR"]==chk_id]['DAYS_EMPLOYED'].iloc[0]
    client_age =InfoClient_test[InfoClient_test["SK_ID_CURR"]==chk_id]['DAYS_BIRTH'].iloc[0]
    client_education = InfoClient_test[InfoClient_test["SK_ID_CURR"]==chk_id]["Education"].iloc[0]
    client_work = InfoClient_test[InfoClient_test["SK_ID_CURR"]==chk_id]["Type de revenu"].iloc[0]

    data_prep = [Genre,Statut_marital,Enfant,Salaires,Montant_du_crédit,Type_credit,Historique_du_crédit,Jours_travaillés,client_age,client_education,client_work]
    data_prep = np.array(data_prep)
    data_prep = data_prep.reshape(1,data_prep.shape[0])
    colonnes = ['Genre','Statut_marital','Enfant','Salaires','Montant_du_crédit','Type_credit','Historique_du_crédit','Nombre d\'années de travail','Age','Education','Type de travail']
    data = pd.DataFrame(data = data_prep, columns = colonnes)
    return data



@st.cache(persist = True)
def load_prediction(X,id):
    #X=sample.iloc[:, :-1]
    score = X[X.index == id].iloc[0]['Score de prediction']
    return score

def threshold(data):
    new_threshold = st.slider(
        label='Threshold:',
        min_value=0.,
        value=0.222,
        max_value=1.)
    preda_proba = data['Score de prediction']
    # new predictions
    pred = (preda_proba >= new_threshold).astype('int')
    # update results
    data['Prediction'] = pred
    return new_threshold, data

html_temp = """
<div style ="background-color:#A2CEDC;padding:13px">
<h1 style ="color:#002366;text-align:center;">Simulateur de prêt</h1>
</div> """

Histoire = """
Home credit est une institution financière internationale non bancaire et se concentre principalement sur les prêts à consommation et aux personnes ayant peu ou pas d’antécédents de crédit.
- Fondé en 1997 en République tchèque et dont le siège est aux Pays-Bas
- La société opère dans 10 pays
- Au 30 juin 2020, le groupe avait servi au total plus de 135,4 millions de clients
- Le principal actionnaire du groupe est PPF,un groupe d’investissement international privé dont le fondateur et principal bénéficiaire était Petr Kellner avec une participation de 88,62% """




Valeurs = """
- Fiabilité
- Transparence
- Professionalisme """

Points_forts= """
- Simplicité
- Rapidité
- Facilité """
#########################################################################################################
# display the front end aspect
st.markdown(html_temp, unsafe_allow_html = True)

data_prediction = pd.read_csv('Data/data_prediction.csv', encoding ='utf-8')
data_test = pd.read_csv('Data/test_selec_boost.csv')
data_test.set_index('SK_ID_CURR', inplace=True)
Info_Client = pd.read_csv('Data/Info_client.csv', index_col='SK_ID_CURR', encoding ='utf-8')
InfoClient_test = pd.read_csv('Data/data_final.csv', encoding ='utf-8')
#InfoClient_test.set_index('SK_ID_CURR', inplace=True)
data_prediction.set_index('SK_ID_CURR', inplace=True)
features = pd.read_csv('Data/HomeCredit_columns_description.csv',encoding='unicode_escape')
id_client = data_test.index.values

st.sidebar.image("Image/bf.png", width=200)
option = st.sidebar.selectbox('Navigation', ('Présentation', 'Caractéristiques des clients',
                                'Prédiction'))



choice = id_client.tolist()
choice.insert(0,'')

chk_id = st.sidebar.selectbox("Veuillez selectioner votre identifiant client :", choice )

if option == 'Présentation':


     st.title("HOME CREDIT")
     st.subheader("Historique :")

     st.markdown(Histoire)

     st.subheader("Principe :")
     st.markdown("Le but de cette application est de mettre en place un système de scoring pour savoir si le client est éligible ou non à un crédit. Cette application va permettre aux clients de comprendre les risques et savoir quels paramètres améliorés pour augmenter ses chances d'obtenir l'accord pour un crédit")
     st.image("Image/canva.png")

     col1, col2 = st.columns(2)
     with col1 :
         st.subheader("Les valeurs de l'entreprise :")
         st.markdown(Valeurs)
     with col2 :
         st.subheader("Les points fors de nos services:")
         st.markdown(Points_forts)

elif option == 'Caractéristiques des clients':

    if chk_id :

        Genre = InfoClient_test[InfoClient_test["SK_ID_CURR"]==chk_id]['Sexe'].iloc[0]
        Statut_marital = InfoClient_test[InfoClient_test["SK_ID_CURR"]==chk_id]['Statut familial'].iloc[0]
        Enfant = InfoClient_test[InfoClient_test["SK_ID_CURR"]==chk_id]['CNT_CHILDREN'].iloc[0]
        Salaires = InfoClient_test[InfoClient_test["SK_ID_CURR"]==chk_id]['AMT_INCOME_TOTAL'].iloc[0]
        Montant_du_crédit = InfoClient_test[InfoClient_test["SK_ID_CURR"]==chk_id]['AMT_CREDIT'].iloc[0]
        Type_credit = InfoClient_test[InfoClient_test["SK_ID_CURR"]==chk_id]["Type du nom du contrat"].iloc[0]
        Historique_du_crédit = InfoClient_test[InfoClient_test["SK_ID_CURR"]==chk_id]['number_prev_loan_extern'].iloc[0]
        Jours_travaillés = InfoClient_test[InfoClient_test["SK_ID_CURR"]==chk_id]['DAYS_EMPLOYED'].iloc[0]
        client_age =InfoClient_test[InfoClient_test["SK_ID_CURR"]==chk_id]['DAYS_BIRTH'].iloc[0]
        client_education = InfoClient_test[InfoClient_test["SK_ID_CURR"]==chk_id]["Education"].iloc[0]
        client_work = InfoClient_test[InfoClient_test["SK_ID_CURR"]==chk_id]["Type de revenu"].iloc[0]
        client_status = InfoClient_test[InfoClient_test["SK_ID_CURR"]==chk_id]["Statut familial"].iloc[0]


        client_age_mean = InfoClient_test['DAYS_BIRTH'].mean()
        client_age_min = InfoClient_test['DAYS_BIRTH'].min()
        client_age_max = InfoClient_test['DAYS_BIRTH'].max()
        client_employed_mean = InfoClient_test['DAYS_EMPLOYED'].mean()
        client_employed_min = InfoClient_test['DAYS_EMPLOYED'].min()
        client_employed_max = InfoClient_test['DAYS_EMPLOYED'].max()
        client_income_mean = InfoClient_test['AMT_INCOME_TOTAL'].mean()
        client_income_min = InfoClient_test['AMT_INCOME_TOTAL'].min()
        client_income_max = InfoClient_test['AMT_INCOME_TOTAL'].max()

        left_column_1, _ = st.columns((3, 2))
        left_column_2, _ = st.columns((3, 1))

            # Age
        if st.sidebar.button("Age"):
            st.sidebar.write("**Age** :", client_age.astype(int), "ans")
            # Graph dans app principale
            if st.sidebar.checkbox("Information age", value = True):
                left_column_1.header("**Age**")
                left_column_1.success("**Vous avez** : **{}** ans".format(client_age.astype(int)))
                fig = go.Figure(go.Indicator(
                    mode="number+gauge+delta",
                    value=client_age.astype(int),
                    delta={
                        'reference': client_age_mean,
                        'increasing': {'color': '#77C5D5'},
                        'decreasing': {'color': '#0093B2'}
                    },
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text':
                        "<b>Age</b><br><span style='color: gray; font-size:0.8em'>mean : 44</span>",
                        'font': {"size": 16}
                    },
                    gauge={'shape': "bullet",
                        'axis': {'range': [client_age_min, client_age_max]},
                        'threshold': {'line': {'color': "red",'width': 2},
                        'thickness': 0.75,
                        'value': client_age
                    },
                        'steps': [{'range': [client_age_min, client_age_mean], 'color': "#0093B2"},
                        {'range': [client_age_mean, client_age_max], 'color': "#B8DDE1"}],
                        'bar': {'color': "#002E5D"}
                    }))
                fig.update_layout(width=700, height=300, margin=dict(l=100, r=50, b=100, t=100, pad=4))
                st.plotly_chart(fig)

        # Code gender
        if st.sidebar.button("Genre"):
            st.sidebar.write("**Genre** :", Genre)
            if st.sidebar.checkbox("Information genre", value = True):
                # Graph dans app principale
                left_column_1.header("**Genre**")
                left_column_1.success("**Vous êtes une/un ** : **{}** ".format(Genre))
                labels = ['Hommes', 'Femmes']
                colors = ['cyan', 'royal blue']
                fig_m = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
                fig_m.add_trace(go.Pie(labels=labels, values=InfoClient_test[InfoClient_test['Prediction'] == 1]['Sexe'].value_counts(), name="Refusé"),
                              1, 1)
                fig_m.add_trace(go.Pie(labels=labels, values=InfoClient_test[InfoClient_test['Prediction'] == 0]['Sexe'].value_counts(), name="Accepté"),
                              1, 2)

                # Use `hole` to create a donut-like pie chart
                fig_m.update_traces(marker=dict(colors=colors), hole=.4, hoverinfo="label+percent+name")

                fig_m.update_layout(
                    title_text="La répartition des prédictions en fonction du genre des clients",
                    # Add annotations in the center of the donut pies.
                    annotations=[dict(text='Refusé', x=0.18, y=0.5, font_size=15, showarrow=False),
                                 dict(text='Accepté', x=0.85, y=0.5, font_size=15, showarrow=False)])
                st.plotly_chart(fig_m)
        # Education
        if st.sidebar.button("Education "):
            st.sidebar.write("**Education** :", client_education)
            if st.sidebar.checkbox(" Information sur les études du client", value = True):
                # Graph dans app principale
                left_column_1.header("**Type d'étude**")
                left_column_2.success("**Votre niveau d'étude** : **{}** ".format(client_education))
                labels = ['Enseignement supérieur', 'Lycée',
               'Enseignement supérieur incomplet', 'Premier cycle de l\'école secondaire', 'Diplome universitaire']
                colors = ['cyan','royal blue', 'blue', 'light blue']
                fig_m = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
                fig_m.add_trace(go.Pie(labels=labels, values=InfoClient_test[InfoClient_test['Prediction'] == 1]['Education'].value_counts(), name="Refusé"),
                              1, 1)
                fig_m.add_trace(go.Pie(labels=labels, values=InfoClient_test[InfoClient_test['Prediction'] == 0]['Education'].value_counts(), name="Accepté"),
                              1, 2)

                # Use `hole` to create a donut-like pie chart
                fig_m.update_traces(marker=dict(colors=colors),hole=.4, hoverinfo="label+percent+name")

                fig_m.update_layout(
                    title_text="La répartition des prédictions en fonction du type d'éducation des clients",
                    # Add annotations in the center of the donut pies.
                    annotations=[dict(text='Refusé', x=0.16, y=0.5, font_size=15, showarrow=False),
                                 dict(text='Accepté', x=0.87, y=0.5, font_size=15, showarrow=False)])
                st.plotly_chart(fig_m)

        # Emploi : durée
        if st.sidebar.button("Années travaillées"):
            st.sidebar.write("**Années travaillées**", Jours_travaillés.astype(int), "ans")
            if st.sidebar.checkbox("Information sur les années travaillées", value = True):
                # Graph dans app principale
                left_column_1.header("**Client : Années travaillées**")
                left_column_1.success("**Le nombre d'années travaillées du client** : **{}** ".format(Jours_travaillés.astype(int)))
                fig = go.Figure(go.Indicator(
                    mode="number+gauge+delta",
                    value=Jours_travaillés.astype(int),
                    delta={
                        'reference': client_employed_mean,
                        'increasing': {'color': '#77C5D5'},
                        'decreasing': {'color': '#0093B2'}
                    },
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text':
                        "<b>Years employed</b><br><span style='color: gray; font-size:0.8em'>mean : 6.4</span>",
                        'font': {"size": 16}
                    },
                    gauge={'shape': "bullet",
                        'axis': {'range': [client_employed_min, client_employed_max]},
                        'threshold': {'line': {'color': "red",'width': 2},
                        'thickness': 0.75,
                        'value': Jours_travaillés.astype(int)
                    },
                    'steps': [{'range': [client_employed_min, client_employed_mean], 'color': "#0093B2"},
                        {'range': [client_employed_mean, client_employed_max], 'color': "#B8DDE1"}],
                        'bar': {'color': "#002E5D"}
                    }))
                fig.update_layout(width=700, height=300, margin=dict(l=180, r=50, b=100, t=100, pad=4))
                st.plotly_chart(fig)

        # Revenus du client
        if st.sidebar.button("Type de revenus"):
            st.sidebar.write("**Type de revenus** :", client_work)
            # Graph dans app principale
            if st.sidebar.checkbox("Information sur le type de revenu", value = True):
                left_column_1.header("**Type de revenus**")
                left_column_1.success("**Le type de revenu du client** : **{}** ".format(client_work))

                labels = ['Working', 'Fonctionnaire de l\'état', 'Pensionaire', 'Associé commercial',
           'Homme d\'affaires', 'Etudiant', 'Chomage']
                colors = ['cyan','royal blue','blue','light blue','9DA2AA','87898A']
                fig_m = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
                fig_m.add_trace(go.Pie(labels=labels, values=InfoClient_test[InfoClient_test['Prediction'] == 1]['Type de revenu'].value_counts(), name="Refusé"),
                              1, 1)
                fig_m.add_trace(go.Pie(labels=labels, values=InfoClient_test[InfoClient_test['Prediction'] == 0]['Type de revenu'].value_counts(), name="Accepté"),
                              1, 2)

                # Use `hole` to create a donut-like pie chart
                fig_m.update_traces(marker=dict(colors=colors),hole=.4, hoverinfo="label+percent+name")

                fig_m.update_layout(
                    title_text="La répartition des prédictions en fonction du type de revenu des clients",
                    # Add annotations in the center of the donut pies.
                    annotations=[dict(text='Refusé', x=0.16, y=0.5, font_size=15, showarrow=False),
                                 dict(text='Accepté', x=0.87, y=0.5, font_size=15, showarrow=False)])
                st.plotly_chart(fig_m)

        if st.sidebar.button("Salaire"):
            st.sidebar.write("**Revenus**", Salaires.astype(int), "$")
             # Graph dans app principale
            if st.sidebar.checkbox("Information salaire", value = True):
                left_column_1.header("**Client : revenus**")
                left_column_1.success("**Votre salaire** : **{}** ".format(Salaires.astype(int)))
                fig = go.Figure(go.Indicator(
                    mode="number+gauge+delta",
                    value=Salaires,
                    delta={
                        'reference': client_income_mean,
                        'increasing': {'color': '#77C5D5'},
                        'decreasing': {'color': '#0093B2'}
                    },
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text':
                        "<b>Income</b><br><span style='color: gray; font-size:0.8em'>mean : 0.17M</span>",
                        'font': {"size": 16}
                    },
                    gauge={'shape': "bullet",
                        'axis': {'range': [client_income_min, client_income_max]},
                        'threshold': {'line': {'color': "red",'width': 2},
                        'thickness': 0.75,
                        'value': Salaires
                    },
                        'steps': [{'range': [client_income_min, client_income_mean], 'color': "#0093B2"},
                        {'range': [client_income_mean, client_income_max], 'color': "#B8DDE1"}],
                        'bar': {'color': "#002E5D"}
                    }))
                fig.update_layout(width=700, height=300, margin=dict(l=110, r=50, b=100, t=100, pad=4))
                st.plotly_chart(fig)

        if st.sidebar.button("Type de contrat"):
            st.sidebar.write("**Type de contrat** :", Type_credit)
            # Graph dans app principale
            if st.sidebar.checkbox("Information type de contrat", value = True):
                left_column_1.header("**Type de contrat**")
                left_column_1.success("**Votre type de contrat** : **{}** ".format(Type_credit))

                labels = ["Crédit cash", "Crédit révolving"]
                fig_m = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
                fig_m.add_trace(go.Pie(labels=labels, values=InfoClient_test[InfoClient_test['Prediction'] == 1]['Type du nom du contrat'].value_counts(), name="Refusé"),
                              1, 1)
                fig_m.add_trace(go.Pie(labels=labels, values=InfoClient_test[InfoClient_test['Prediction'] == 0]['Type du nom du contrat'].value_counts(), name="Accepté"),
                              1, 2)

                # Use `hole` to create a donut-like pie chart
                fig_m.update_traces(hole=.4, hoverinfo="label+percent+name")

                fig_m.update_layout(
                    title_text="La répartition des prédictions en fonction du type de crédit des clients",
                    # Add annotations in the center of the donut pies.
                    annotations=[dict(text='Refusé', x=0.16, y=0.5, font_size=20, showarrow=False),
                                 dict(text='Accepté', x=0.87, y=0.5, font_size=20, showarrow=False)])
                st.plotly_chart(fig_m)

elif option == 'Prédiction' :
    if chk_id :

        st.caption("Récapitulatif des caractéristiques principales du client : ")

        data= get_carac(chk_id)

        st.dataframe(data.style.hide_index().set_properties(**{'border': '1.3px solid black',
                          'color': "#248FAF"}))
        features.drop( features[ features['Row'] == 'SK_ID_CURR' ].index, inplace=True)
        features.drop( features[ features['Row'] == 'TARGET' ].index, inplace=True)
        liste_features = features.Row.tolist()
        liste_features.insert(0,'')
        var = st.sidebar.selectbox('Choisissez une variable caractéristique parmi l\'ensemble de test pour avoir son explication (en anglais) :'  , liste_features )
        if var :
            st.sidebar.info(features[features['Row'] == var]['Description'].values)
        pm = st.button('Performance du modèle')
        if pm:
            st.metric('Efficacité du modèle', "77%")
            st.caption("77% des données ont bien été classifié")
            st.image("Image/roc_model_final.png")
            st.markdown("Le modèle est à **77 %** fiable. La figure ci_dessus nous indique la matrice de confusion et la courbe ROC.")
            st.caption("La matrice de confusion permet de résumer les performances de la classification de nos données. Les cases diagonales sont les données bien classifiées dans chacune des catégories, leur nombre doivent être la plus élevée possible. Les deux autres cases résument les données mal classifiées et leur nombre doivent être le plus bas possible.")
        sp = st.button('Score de prédiction')
        if sp :
            prediction = load_prediction(data_prediction, chk_id)
            new_threshold, data_prediction = threshold(data_prediction)
            # pred_client = data_prediction.iloc[1]['Prediction']
            import json
            # url_get = "http://127.0.0.1:8000/"
            # get = requests.get(url_get)
            # st.write(get)
            url = "http://fastapi:8000/predict"
            data={"id":chk_id}
            pred_client = requests.post(url, json=data)
            st.write("la classe est : ",pred_client.json())
            st.spinner('Chargement du score du client...')
            st.write("La probabilité de defaut de paiement: {:.0f}%".format(round(float(prediction), 2)))
            gauge_score = round(float(prediction))/100
            ########################
            gauge = go.Figure(go.Indicator(
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Score de prédiction", 'font': {'size': 24}},
            value = gauge_score,
            mode = "gauge+delta+number",

            gauge = {'axis': {'range': [None, 1]},
                     'steps' : [
                         {'range': [0, 0.2], 'color': "#008177"},
                         {'range': [0.2, 0.4], 'color': "#00BAB3"},
                         {'range': [0.4, 0.5], 'color': "#D4E88B"},
                         {'range': [0.5, 0.6], 'color': "#F4EA9D"},
                         {'range': [0.6, 0.8], 'color': "#FF9966"},
                         {'range': [0.8, 1], 'color': "#E2383F"},
                     ],
                     'threshold': {
                    'line': {'color': "black", 'width': 10},
                    'thickness': 0.8,
                    'value': gauge_score},

                     'bar': {'color': "black", 'thickness' : 0.15},
                    },
            ))

            gauge.update_layout(width=600, height=500,
                        margin=dict(l=50, r=50, b=100, t=100, pad=4))

            st.plotly_chart(gauge)

            if 0< gauge_score <=0.222 :
                st.success('Vous êtes éligible !')
            else :
                st.error('Vous n\'êtes pas éligible')

        cc = st.button("comparaison avec d'autres clients")

        if cc:
            st.write('Comparaison de votre score à la moyenne des scores des 20 clients les plus similaires à votre profil:')
            InfoClient_test_index_reset = InfoClient_test.set_index('SK_ID_CURR')
            fa = InfoClient_test_index_reset[["Score de prediction","PREDICTION_NEIGHBORS_20_MEAN"]].loc[chk_id]
            value = fa.values
            fig = px.bar(fa, x=["Score de prediction","PREDICTION_NEIGHBORS_20_MEAN"], y=value)
            fig.update_layout(
             margin=dict(l=20, r=20, t=20, b=20),paper_bgcolor="#A2CEDC", plot_bgcolor='white')
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#A2CEDC")
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#A2CEDC")
            st.plotly_chart(fig)
        interp = st.button("Interprétabilité")
        if interp:

            st.components.v1.iframe("http://HERON:8050/", width =1200, height= 1000, scrolling = True)


        st.session_state.update = st.button("Non éligible, cliquez ici!")

        if st.session_state.update:
            features_name = data_test.columns.tolist()
            features_name.insert(0,'')

            features_choice = st.selectbox("Choisissez une variable à modifier :", features_name)
            if features_choice :
             #filtre id client ligne du client dans la data test

                client = data_test[data_test.index == chk_id]
                  #Valeur originale de la variable choisis
                original_value = float(client[features_choice].values[0])
                if (float(data_test[features_choice].min()) , float(data_test[features_choice].max())) == (0,1):
                    step = float(1)
                else :
                    step = ((float(data_test[features_choice].min()) - float(data_test[features_choice].max()))  / 20)

                new_value = st.slider('Changer la valeur :', min_value = float(data_test[features_choice].min()) , max_value = float(data_test[features_choice].max()), value = original_value, step=step)
                st.write("valeur", new_value)
                client[features_choice].values[0] = new_value
                new_proba_pred = clf.predict_proba(client)[:,1]
                 #new_threshold, data_prediction = threshold(data_prediction)
                if new_proba_pred > 0.22:
                    st.write(new_proba_pred)
                    st.error("Vous êtes à nouveau refusé ! ")
                else :
                    st.write(new_proba_pred)
                    st.success("Vous êtes accepté !")

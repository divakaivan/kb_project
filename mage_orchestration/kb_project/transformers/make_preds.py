# import mlflow

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
import numpy as np

from typing import Dict, List
from datetime import datetime

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

######## GCN ########
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(18, 64)
        self.conv2 = GCNConv(64,32)
        self.conv3 = GCNConv(32,16)
        self.conv4 = GCNConv(16,2)

    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)

        return F.log_softmax(x, dim=1)

model = GCN()
model_path = 'models_gcn_20240728_050159/gcn.pth' # can replace with mlflow if setup
model.load_state_dict(torch.load(model_path))
model.eval()


cols_for_models = ['amt', 'is_fraud', 'category_food_dining', 'category_gas_transport', 'category_grocery_net', 'category_grocery_pos', 'category_health_fitness', 'category_home', 'category_kids_pets', 'category_misc_net', 'category_misc_pos', 'category_personal_care', 'category_shopping_net', 'category_shopping_pos', 'category_travel', 'state_AL', 'state_AR', 'state_AZ', 'state_CA', 'state_CO', 'state_CT', 'state_DC', 'state_DE', 'state_FL', 'state_GA', 'state_HI', 'state_IA', 'state_ID', 'state_IL', 'state_IN', 'state_KS', 'state_KY', 'state_LA', 'state_MA', 'state_MD', 'state_ME', 'state_MI', 'state_MN', 'state_MO', 'state_MS', 'state_MT', 'state_NC', 'state_ND', 'state_NE', 'state_NH', 'state_NJ', 'state_NM', 'state_NV', 'state_NY', 'state_OH', 'state_OK', 'state_OR', 'state_PA', 'state_RI', 'state_SC', 'state_SD', 'state_TN', 'state_TX', 'state_UT', 'state_VA', 'state_VT', 'state_WA', 'state_WI', 'state_WV', 'state_WY', 'job_Accountant, chartered', 'job_Accountant, chartered certified', 'job_Accountant, chartered public finance', 'job_Accounting technician', 'job_Acupuncturist', 'job_Administrator', 'job_Administrator, arts', 'job_Administrator, charities/voluntary organisations', 'job_Administrator, education', 'job_Administrator, local government', 'job_Advertising account executive', 'job_Advertising account planner', 'job_Advertising copywriter', 'job_Advice worker', 'job_Aeronautical engineer', 'job_Agricultural consultant', 'job_Aid worker', 'job_Air broker', 'job_Air cabin crew', 'job_Air traffic controller', 'job_Airline pilot', 'job_Ambulance person', 'job_Amenity horticulturist', 'job_Analytical chemist', 'job_Animal nutritionist', 'job_Animal technologist', 'job_Animator', 'job_Applications developer', 'job_Arboriculturist', 'job_Archaeologist', 'job_Architect', 'job_Architectural technologist', 'job_Archivist', 'job_Armed forces logistics/support/administrative officer', 'job_Armed forces technical officer', 'job_Armed forces training and education officer', 'job_Art gallery manager', 'job_Art therapist', 'job_Artist', 'job_Arts development officer', 'job_Associate Professor', 'job_Audiological scientist', 'job_Barista', 'job_Barrister', "job_Barrister's clerk", 'job_Biochemist, clinical', 'job_Biomedical engineer', 'job_Biomedical scientist', 'job_Bookseller', 'job_Broadcast engineer', 'job_Broadcast journalist', 'job_Broadcast presenter', 'job_Building control surveyor', 'job_Building services engineer', 'job_Building surveyor', 'job_Buyer, industrial', 'job_Buyer, retail', 'job_Cabin crew', 'job_Call centre manager', 'job_Camera operator', 'job_Careers adviser', 'job_Careers information officer', 'job_Cartographer', 'job_Catering manager', 'job_Ceramics designer', 'job_Charity fundraiser', 'job_Charity officer', 'job_Chartered accountant', 'job_Chartered legal executive (England and Wales)', 'job_Chartered loss adjuster', 'job_Chartered public finance accountant', 'job_Chemical engineer', 'job_Chemist, analytical', 'job_Chief Executive Officer', 'job_Chief Financial Officer', 'job_Chief Marketing Officer', 'job_Chief Operating Officer', 'job_Chief Strategy Officer', 'job_Chief Technology Officer', 'job_Chief of Staff', 'job_Child psychotherapist', 'job_Chiropodist', 'job_Civil Service administrator', 'job_Civil Service fast streamer', 'job_Civil engineer, contracting', 'job_Claims inspector/assessor', 'job_Clinical biochemist', 'job_Clinical cytogeneticist', 'job_Clinical psychologist', 'job_Clinical research associate', 'job_Clothing/textile technologist', 'job_Colour technologist', 'job_Commercial horticulturist', 'job_Commercial/residential surveyor', 'job_Commissioning editor', 'job_Communications engineer', 'job_Community arts worker', 'job_Community development worker', 'job_Community education officer', 'job_Community pharmacist', 'job_Company secretary', 'job_Comptroller', 'job_Conservation officer, historic buildings', 'job_Conservator, furniture', 'job_Conservator, museum/gallery', 'job_Contracting civil engineer', 'job_Contractor', 'job_Control and instrumentation engineer', 'job_Copy', 'job_Copywriter, advertising', 'job_Corporate investment banker', 'job_Counselling psychologist', 'job_Counsellor', 'job_Curator', 'job_Cytogeneticist', 'job_Dance movement psychotherapist', 'job_Dancer', 'job_Data scientist', 'job_Database administrator', 'job_Dealer', 'job_Designer, ceramics/pottery', 'job_Designer, exhibition/display', 'job_Designer, furniture', 'job_Designer, industrial/product', 'job_Designer, interior/spatial', 'job_Designer, jewellery', 'job_Designer, multimedia', 'job_Designer, television/film set', 'job_Designer, textile', 'job_Development worker, community', 'job_Development worker, international aid', 'job_Diagnostic radiographer', 'job_Dispensing optician', 'job_Doctor, general practice', 'job_Doctor, hospital', 'job_Drilling engineer', 'job_Early years teacher', 'job_Ecologist', 'job_Economist', 'job_Editor, commissioning', 'job_Editor, film/video', 'job_Editor, magazine features', 'job_Education administrator', 'job_Education officer, community', 'job_Education officer, museum', 'job_Educational psychologist', 'job_Electrical engineer', 'job_Electronics engineer', 'job_Embryologist, clinical', 'job_Emergency planning/management officer', 'job_Energy engineer', 'job_Energy manager', 'job_Engineer, aeronautical', 'job_Engineer, agricultural', 'job_Engineer, automotive', 'job_Engineer, biomedical', 'job_Engineer, broadcasting (operations)', 'job_Engineer, building services', 'job_Engineer, civil (consulting)', 'job_Engineer, civil (contracting)', 'job_Engineer, communications', 'job_Engineer, control and instrumentation', 'job_Engineer, drilling', 'job_Engineer, electronics', 'job_Engineer, land', 'job_Engineer, maintenance', 'job_Engineer, manufacturing', 'job_Engineer, materials', 'job_Engineer, mining', 'job_Engineer, petroleum', 'job_Engineer, production', 'job_Engineer, site', 'job_Engineer, structural', 'job_Engineer, technical sales', 'job_Engineering geologist', 'job_English as a foreign language teacher', 'job_English as a second language teacher', 'job_Environmental consultant', 'job_Environmental education officer', 'job_Environmental health practitioner', 'job_Environmental manager', 'job_Equality and diversity officer', 'job_Equities trader', 'job_Estate manager/land agent', 'job_Event organiser', 'job_Exercise physiologist', 'job_Exhibition designer', 'job_Exhibitions officer, museum/gallery', 'job_Facilities manager', 'job_Farm manager', 'job_Field seismologist', 'job_Field trials officer', 'job_Film/video editor', 'job_Financial adviser', 'job_Financial trader', 'job_Fine artist', 'job_Firefighter', 'job_Fisheries officer', 'job_Fitness centre manager', 'job_Forensic psychologist', 'job_Forest/woodland manager', 'job_Freight forwarder', 'job_Furniture conservator/restorer', 'job_Furniture designer', 'job_Further education lecturer', 'job_Futures trader', 'job_Gaffer', 'job_Garment/textile technologist', 'job_General practice doctor', 'job_Geneticist, molecular', 'job_Geochemist', 'job_Geologist, engineering', 'job_Geologist, wellsite', 'job_Geophysicist/field seismologist', 'job_Geoscientist', 'job_Glass blower/designer', 'job_Health and safety adviser', 'job_Health physicist', 'job_Health promotion specialist', 'job_Health service manager', 'job_Health visitor', 'job_Herbalist', 'job_Heritage manager', 'job_Herpetologist', 'job_Higher education careers adviser', 'job_Historic buildings inspector/conservation officer', 'job_Homeopath', 'job_Horticultural consultant', 'job_Horticultural therapist', 'job_Horticulturist, commercial', 'job_Hospital doctor', 'job_Hospital pharmacist', 'job_Hotel manager', 'job_Human resources officer', 'job_Hydrogeologist', 'job_Hydrographic surveyor', 'job_Hydrologist', 'job_IT consultant', 'job_IT trainer', 'job_Illustrator', 'job_Immigration officer', 'job_Immunologist', 'job_Industrial buyer', 'job_Industrial/product designer', 'job_Information officer', 'job_Information systems manager', 'job_Insurance broker', 'job_Insurance claims handler', 'job_Insurance risk surveyor', 'job_Insurance underwriter', 'job_Intelligence analyst', 'job_Interior and spatial designer', 'job_Interpreter', 'job_Investment analyst', 'job_Investment banker, corporate', 'job_Investment banker, operational', 'job_Jewellery designer', 'job_Journalist, newspaper', 'job_Land', 'job_Land/geomatics surveyor', 'job_Landscape architect', 'job_Lawyer', 'job_Learning disability nurse', 'job_Learning mentor', 'job_Lecturer, further education', 'job_Lecturer, higher education', 'job_Legal secretary', 'job_Leisure centre manager', 'job_Lexicographer', 'job_Librarian, academic', 'job_Librarian, public', 'job_Licensed conveyancer', 'job_Local government officer', 'job_Location manager', 'job_Logistics and distribution manager', 'job_Loss adjuster, chartered', 'job_Magazine features editor', 'job_Magazine journalist', 'job_Maintenance engineer', 'job_Make', 'job_Management consultant', 'job_Manufacturing engineer', 'job_Manufacturing systems engineer', 'job_Market researcher', 'job_Marketing executive', 'job_Materials engineer', 'job_Mechanical engineer', 'job_Media buyer', 'job_Media planner', 'job_Medical physicist', 'job_Medical sales representative', 'job_Medical secretary', 'job_Medical technical officer', 'job_Mental health nurse', 'job_Merchandiser, retail', 'job_Metallurgist', 'job_Minerals surveyor', 'job_Mining engineer', 'job_Mudlogger', 'job_Multimedia programmer', 'job_Museum education officer', 'job_Museum/gallery conservator', 'job_Museum/gallery exhibitions officer', 'job_Music therapist', 'job_Music tutor', 'job_Musician', 'job_Nature conservation officer', 'job_Naval architect', 'job_Network engineer', 'job_Neurosurgeon', "job_Nurse, children's", 'job_Nurse, mental health', 'job_Nutritional therapist', 'job_Occupational hygienist', 'job_Occupational psychologist', 'job_Occupational therapist', 'job_Oceanographer', 'job_Oncologist', 'job_Operational researcher', 'job_Operations geologist', 'job_Optician, dispensing', 'job_Optometrist', 'job_Orthoptist', 'job_Osteopath', 'job_Outdoor activities/education manager', 'job_Paediatric nurse', 'job_Paramedic', 'job_Patent attorney', 'job_Pathologist', 'job_Pension scheme manager', 'job_Pensions consultant', 'job_Personnel officer', 'job_Petroleum engineer', 'job_Pharmacist, community', 'job_Pharmacist, hospital', 'job_Pharmacologist', 'job_Physicist, medical', 'job_Physiological scientist', 'job_Physiotherapist', 'job_Phytotherapist', 'job_Pilot, airline', 'job_Planning and development surveyor', 'job_Plant breeder/geneticist', 'job_Podiatrist', 'job_Police officer', "job_Politician's assistant", 'job_Presenter, broadcasting', 'job_Press photographer', 'job_Press sub', 'job_Primary school teacher', 'job_Prison officer', 'job_Private music teacher', 'job_Probation officer', 'job_Producer, radio', 'job_Producer, television/film/video', 'job_Product designer', 'job_Product manager', 'job_Product/process development scientist', 'job_Production assistant, radio', 'job_Production assistant, television', 'job_Production engineer', 'job_Production manager', 'job_Professor Emeritus', 'job_Programme researcher, broadcasting/film/video', 'job_Programmer, applications', 'job_Programmer, multimedia', 'job_Psychiatric nurse', 'job_Psychiatrist', 'job_Psychologist, clinical', 'job_Psychologist, counselling', 'job_Psychologist, forensic', 'job_Psychologist, sport and exercise', 'job_Psychotherapist', 'job_Psychotherapist, child', 'job_Public affairs consultant', 'job_Public house manager', 'job_Public librarian', 'job_Public relations account executive', 'job_Public relations officer', 'job_Purchasing manager', 'job_Quantity surveyor', 'job_Quarry manager', 'job_Race relations officer', 'job_Radio broadcast assistant', 'job_Radio producer', 'job_Radiographer, diagnostic', 'job_Radiographer, therapeutic', 'job_Records manager', 'job_Regulatory affairs officer', 'job_Research officer, political party', 'job_Research officer, trade union', 'job_Research scientist (life sciences)', 'job_Research scientist (maths)', 'job_Research scientist (medical)', 'job_Research scientist (physical sciences)', 'job_Restaurant manager, fast food', 'job_Retail banker', 'job_Retail buyer', 'job_Retail manager', 'job_Retail merchandiser', 'job_Risk analyst', 'job_Rural practice surveyor', 'job_Sales executive', 'job_Sales professional, IT', 'job_Sales promotion account executive', 'job_Science writer', 'job_Scientific laboratory technician', 'job_Scientist, audiological', 'job_Scientist, biomedical', 'job_Scientist, clinical (histocompatibility and immunogenetics)', 'job_Scientist, marine', 'job_Scientist, physiological', 'job_Scientist, research (maths)', 'job_Scientist, research (medical)', 'job_Scientist, research (physical sciences)', 'job_Secondary school teacher', 'job_Secretary/administrator', 'job_Seismic interpreter', 'job_Senior tax professional/tax inspector', 'job_Set designer', 'job_Ship broker', 'job_Site engineer', 'job_Social research officer, government', 'job_Social researcher', 'job_Soil scientist', 'job_Solicitor', 'job_Solicitor, Scotland', 'job_Special educational needs teacher', 'job_Special effects artist', 'job_Sport and exercise psychologist', 'job_Sports administrator', 'job_Sports development officer', 'job_Stage manager', 'job_Statistician', 'job_Structural engineer', 'job_Sub', 'job_Surgeon', 'job_Surveyor, hydrographic', 'job_Surveyor, land/geomatics', 'job_Surveyor, minerals', 'job_Surveyor, mining', 'job_Surveyor, rural practice', 'job_Systems analyst', 'job_Systems developer', 'job_TEFL teacher', 'job_Tax adviser', 'job_Tax inspector', 'job_Teacher, English as a foreign language', 'job_Teacher, adult education', 'job_Teacher, early years/pre', 'job_Teacher, primary school', 'job_Teacher, secondary school', 'job_Teacher, special educational needs', 'job_Teaching laboratory technician', 'job_Technical brewer', 'job_Telecommunications researcher', 'job_Television camera operator', 'job_Television floor manager', 'job_Television production assistant', 'job_Television/film/video producer', 'job_Textile designer', 'job_Theatre director', 'job_Theatre manager', 'job_Theme park manager', 'job_Therapist, art', 'job_Therapist, drama', 'job_Therapist, horticultural', 'job_Therapist, music', 'job_Therapist, occupational', 'job_Therapist, sports', 'job_Tour manager', 'job_Tourism officer', 'job_Tourist information centre manager', 'job_Town planner', 'job_Toxicologist', 'job_Trade mark attorney', 'job_Trading standards officer', 'job_Training and development officer', 'job_Transport planner', 'job_Travel agency manager', 'job_Tree surgeon', 'job_Veterinary surgeon', 'job_Video editor', 'job_Visual merchandiser', 'job_Volunteer coordinator', 'job_Warden/ranger', 'job_Warehouse manager', 'job_Waste management officer', 'job_Water engineer', 'job_Water quality scientist', 'job_Web designer', 'job_Wellsite geologist', 'job_Writer', 'trans_hour_1', 'trans_hour_2', 'trans_hour_3', 'trans_hour_4', 'trans_hour_5', 'trans_hour_6', 'trans_hour_7', 'trans_hour_8', 'trans_hour_9', 'trans_hour_10', 'trans_hour_11', 'trans_hour_12', 'trans_hour_13', 'trans_hour_14', 'trans_hour_15', 'trans_hour_16', 'trans_hour_17', 'trans_hour_18', 'trans_hour_19', 'trans_hour_20', 'trans_hour_21', 'trans_hour_22', 'trans_hour_23', 'trans_dow_1', 'trans_dow_2', 'trans_dow_3', 'trans_dow_4', 'trans_dow_5', 'trans_dow_6']
######## XGBoost ########
xgboost_model_path = 'models_xgboost_20240804_005923/xgb_model.json' 
xgb_model = xgb.Booster()
xgb_model.load_model(xgboost_model_path)

######## CatBoost ########
catboost_model_path = 'models_catboost_20240804_004758/catboost_model.cbm' 
catboost_model = CatBoostClassifier()
catboost_model.load_model(catboost_model_path)

@transformer
def transform(messages: List[Dict], *args, **kwargs):

    for msg in messages:
        msg['merchant'] = msg['merchant'].replace('fraud_', '')

        # GCN prediction
        numerical_cols = ['lat', 'long', 'amt', 'merch_lat', 'merch_long']
        categories = {
            'food_dining': 0, 'gas_transport': 0, 'grocery_net': 0, 'grocery_pos': 0,
            'health_fitness': 0, 'home': 0, 'kids_pets': 0, 'misc_net': 0, 'misc_pos': 0,
            'personal_care': 0, 'shopping_net': 0, 'shopping_pos': 0, 'travel': 0
        }

        category = msg.get('category')
        try:
            categories[category] += 1
        except KeyError:
            pass
        
        numericals_data = [msg.get(num) for num in numerical_cols]
        data_inputs = numericals_data + list(categories.values())
        data_inputs = torch.tensor(data_inputs, dtype=torch.float32).unsqueeze(0)
        empty_edge_index = torch.empty((2, 0), dtype=torch.long) 
        single_test_data = Data(x=data_inputs, edge_index=empty_edge_index)
        with torch.no_grad():
            prediction = model(single_test_data.x, single_test_data.edge_index)

        predicted_class = prediction.argmax(dim=1).item()
        msg['pred_gcn_is_fraud'] = predicted_class


        ######## Prepare data for XGBoost and CatBoost ########
        df = pd.DataFrame(msg, index=[0])
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['trans_hour'] = df['trans_date_trans_time'].dt.hour
        df['trans_dow'] = df['trans_date_trans_time'].dt.dayofweek
    
        categorical_features = [
            'category', 'state',
            'job', 'trans_hour', 'trans_dow'
            ]  
        df_cat = df[categorical_features]

        cat_ohe = {s: 0 for s in cols_for_models}
        for _, row in df_cat.iterrows():
            for feature in df_cat.columns:
                key = f"{feature}_{row[feature]}"
                cat_ohe[key] = 1

        ready_df = pd.DataFrame(cat_ohe, index=[0])
        ready_df['amt'] = msg.get('amt')
        ready_df.drop('is_fraud', axis=1, inplace=True)

        # train data has dummies and drop_first is used
        # this is to remove those cols if they exist
        try:
            ready_df.drop('category_entertainment', axis=1, inplace=True)
        except KeyError:
            pass
        try:
            ready_df.drop('trans_hour_0', axis=1, inplace=True)
        except KeyError:
            pass
        try:
            ready_df.drop('state_AK', axis=1, inplace=True)
        except KeyError:
            pass
        try:
            ready_df.drop('trans_dow_0', axis=1, inplace=True)
        except KeyError:
            pass
        try:
            ready_df.drop('job_Academic librarian', axis=1, inplace=True)
        except KeyError:
            pass 

        ######## XGBoost ########
        dmatrix = xgb.DMatrix(ready_df)
        xgb_pred = xgb_model.predict(dmatrix)
        predicted_class_xgb = int(xgb_pred.argmax())
        msg['pred_xgb_is_fraud'] = predicted_class_xgb

        ######## CatBoost ########
        catboost_pred = catboost_model.predict(ready_df)
        predicted_class_catboost = int(catboost_pred[0])
        msg['pred_catboost_is_fraud'] = predicted_class_catboost

        # fake date as if transactions are recent
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        msg['trans_date_trans_time'] = now

        if msg['pred_xgb_is_fraud'] == 1 and msg['pred_catboost_is_fraud'] == 1 and msg['pred_gcn_is_fraud'] == 1:
                kakao_msg = (
                    f"안녕하세요.\n"
                    f"KB 국민은행 사기 탐지 팀입니다. 고객님의 계좌에서 의심스러운 거래가 발생했음을 알려드립니다.\n\n"
                    f"일시: {msg['trans_date_trans_time']}\n"
                    f"금액: {msg['amt']}$ ({format(int(round(msg['amt'] * 1339.44, 0)), ',')}원)\n"
                    f"상점: {msg['merchant']}\n\n"
                    f"거래를 하지 않으셨다면 즉시 \n[국내] : (지역번호 없이) 1588-9999, 1599-9999, 1644-9999 \n[해외] : 02) 6300-9999 (82-2-6300-9999) 연락하시거나, 저희 모바일 앱을 통해 연락해 주시기 바랍니다.\n"
                    f"이 거래를 인지하셨다면 이 메시지를 무시하셔도 됩니다."
                )

                import requests
                import json
                
                url = "https://kapi.kakao.com/v1/api/talk/friends/message/default/send"
                # Requires kakao api application to be set up
                headers={
                    "Authorization" : "Bearer " + kwargs['kakao_token']
                }

                data={
                    "receiver_uuids": json.dumps([kwargs['user_id']]),
                    "template_object": json.dumps({
                        "object_type": "text",
                        "text": kakao_msg,
                        "link": {
                            "web_url": "https://obank.kbstar.com/quics?page=C023171#loading"
                        }
                    })
                }

                response = requests.post(url, headers=headers, data=data)

                print('<<<[이상거래 탐지 알림] 고객님께 알림을 보내 드렸습니다>>>')
            
    return messages

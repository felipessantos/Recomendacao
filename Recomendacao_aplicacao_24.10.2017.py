
# coding: utf-8

# # Treinando RandomForest

# In[1]:

import pandas as pd
from datetime import datetime, timedelta, date
import numpy as np
import csv
import unidecode 
import pandas.core.algorithms as algos
from scipy.stats import kendalltau   
from funcoes_uteis import *
from dateutil.relativedelta import relativedelta

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

import string


# In[2]:

def diff_month(d1, d2):
    return (d1.year - d2.year)*12 + d1.month - d2.month

def periodicidade(x):
    if x == 'Trienal':
        return 36    
    elif x == 'Anual':
        return 12
    elif x == 'Semestral':
        return 6
    elif x == 'Trimestral':
        return 3
    else: 
        return 1

def marca_base(Perc, x):
    if x >= Perc[(len(Perc)-1)]:
        return len(Perc) +1
    else:
        for i in range(len(Perc)):
            if x < Perc[i]:
                return i + 1

def cria_curva(percentiles, variavel):
    Perc = list()
    for i in range(len(percentiles)):
        Perc.append(np.percentile(variavel, percentiles[i]))
    return Perc    

def contagens_anteriores(um_df):
    um_df['contagens_anteriores'] = range(len(um_df))
    return um_df


# In[3]:

def cria_chave(df_recomendacao):
    if len(df_recomendacao.columns) == 3:
        df_recomendacao['chave_recomendacao'] = df_recomendacao.Produto1
        df_recomendacao['ultimo_produto'] = df_recomendacao.Produto2
        return df_recomendacao
    elif len(df_recomendacao.columns) == 4:
        df_recomendacao['chave_recomendacao'] = df_recomendacao.Produto1+df_recomendacao.Produto2
        df_recomendacao['ultimo_produto'] = df_recomendacao.Produto3
        return df_recomendacao
    elif len(df_recomendacao.columns) == 5:
        df_recomendacao['chave_recomendacao'] = df_recomendacao.Produto1+df_recomendacao.Produto2+df_recomendacao.Produto3    
        df_recomendacao['ultimo_produto'] = df_recomendacao.Produto4
        return df_recomendacao
    elif len(df_recomendacao.columns) == 6:
        df_recomendacao['chave_recomendacao'] = df_recomendacao.Produto1+df_recomendacao.Produto2+df_recomendacao.Produto3+df_recomendacao.Produto4
        df_recomendacao['ultimo_produto'] = df_recomendacao.Produto5
        return df_recomendacao
    elif len(df_recomendacao.columns) == 7:
        df_recomendacao['chave_recomendacao'] = df_recomendacao.Produto1+df_recomendacao.Produto2+df_recomendacao.Produto3+df_recomendacao.Produto4+df_recomendacao.Produto5
        df_recomendacao['ultimo_produto'] = df_recomendacao.Produto6     
        return df_recomendacao
    elif len(df_recomendacao.columns) == 8:
        df_recomendacao['chave_recomendacao'] = df_recomendacao.Produto1+df_recomendacao.Produto2+df_recomendacao.Produto3+df_recomendacao.Produto4+df_recomendacao.Produto5+df_recomendacao.Produto6
        df_recomendacao['ultimo_produto'] = df_recomendacao.Produto7       
        return df_recomendacao 
    elif len(df_recomendacao.columns) == 9:
        df_recomendacao['chave_recomendacao'] = df_recomendacao.Produto1+df_recomendacao.Produto2+df_recomendacao.Produto3+df_recomendacao.Produto4+df_recomendacao.Produto5+df_recomendacao.Produto6+df_recomendacao.Produto7
        df_recomendacao['ultimo_produto'] = df_recomendacao.Produto8       
        return df_recomendacao 
    elif len(df_recomendacao.columns) == 10:
        df_recomendacao['chave_recomendacao'] = df_recomendacao.Produto1+df_recomendacao.Produto2+df_recomendacao.Produto3+df_recomendacao.Produto4+df_recomendacao.Produto5+df_recomendacao.Produto6+df_recomendacao.Produto7+df_recomendacao.Produto8
        df_recomendacao['ultimo_produto'] = df_recomendacao.Produto9       
        return df_recomendacao 
    elif len(df_recomendacao.columns) == 11:
        df_recomendacao['chave_recomendacao'] = df_recomendacao.Produto1+df_recomendacao.Produto2+df_recomendacao.Produto3+df_recomendacao.Produto4+df_recomendacao.Produto5+df_recomendacao.Produto6+df_recomendacao.Produto7+df_recomendacao.Produto8+df_recomendacao.Produto9
        df_recomendacao['ultimo_produto'] = df_recomendacao.Produto10       
        return df_recomendacao 
    elif len(df_recomendacao.columns) == 12:
        df_recomendacao['chave_recomendacao'] = df_recomendacao.Produto1+df_recomendacao.Produto2+df_recomendacao.Produto3+df_recomendacao.Produto4+df_recomendacao.Produto5+df_recomendacao.Produto6+df_recomendacao.Produto7+df_recomendacao.Produto8+df_recomendacao.Produto9+df_recomendacao.Produto10
        df_recomendacao['ultimo_produto'] = df_recomendacao.Produto11        
        return df_recomendacao 


# In[4]:

def Recomendacao_Base1(fim_janela_feature, janela_booking, df_base):
    aux_janela_feature = fim_janela_feature+ relativedelta(months=-12)
    inicio_janela_booking = fim_janela_feature
    fim_janela_booking = fim_janela_feature+ relativedelta(months=janela_booking)

    df_booking = df_base[(df_base.Instalacao_AnoMes >= inicio_janela_booking) & 
                         (df_base.Instalacao_AnoMes < fim_janela_booking)].copy()
    df_booking = df_booking[~df_booking.Servico.isnull()].copy()
    lista = list(df_booking.Servico.unique())
    lista = ['Comprou_' + str(i) for i in lista]
    dict_lista = {str(i): 'sum' for i in lista}
    df_booking.rename(columns= {'Servico': 'Comprou'}, inplace= True)
    ohe = ['Comprou']
    colunas = ['cd_ChaveCliente', 'Comprou']
    df_ohe_Servico = pd.get_dummies(df_booking[colunas], columns = ohe,)
    df_ohe_Servico = df_ohe_Servico.groupby('cd_ChaveCliente').agg(dict_lista)
    return df_ohe_Servico   


# In[5]:

def calcula_recomendacao(df_recomendacao, chave):
    dict_lista = {'cd_ChaveCliente' : 'sum',}
    colunas_aux = ['ultimo_produto', 'chave_recomendacao']
    df_recomendacao1 = df_recomendacao.reset_index().groupby(colunas_aux).agg(dict_lista)
    df_recomendacao1.sort_values(['cd_ChaveCliente'], ascending= [0] ,inplace=True)
    df_recomendacao1.reset_index(inplace= True)
    
    df_recomendacao2 = df_recomendacao1[df_recomendacao1.chave_recomendacao != chave].copy()
    df_recomendacao2['chave_recomendacao'] = chave
    dict_lista = {'cd_ChaveCliente' : 'sum',}
    df_recomendacao2 = df_recomendacao2.reset_index().groupby(colunas_aux).agg(dict_lista)
    df_recomendacao2.sort_values(['cd_ChaveCliente'], ascending= [0] ,inplace=True)
    df_recomendacao2.reset_index(inplace= True)
    df_recomendacao2['perc_complementar'] = df_recomendacao2.cd_ChaveCliente*100/df_recomendacao2.cd_ChaveCliente.sum()
    
    df_recomendacao1 = df_recomendacao1[df_recomendacao1.chave_recomendacao == chave].copy()
    df_recomendacao1['perc'] = df_recomendacao1.cd_ChaveCliente*100/df_recomendacao1.cd_ChaveCliente.sum()
    
    aux = ['ultimo_produto','chave_recomendacao','perc_complementar']
    df_recomendacao_aux = pd.merge(df_recomendacao1, df_recomendacao2[aux], on=['chave_recomendacao', 'ultimo_produto'])
    df_recomendacao_aux['ratio_recomenda'] = df_recomendacao_aux.perc/df_recomendacao_aux.perc_complementar
    
    aux = ['ratio_recomenda','perc']
        
    RatioCorrigido = df_recomendacao_aux[df_recomendacao_aux.ratio_recomenda >= 1].copy()
    RatioCorrigido.sort_values(['ratio_recomenda'], ascending= [0] ,inplace=True)
        
    Freqencia = df_recomendacao_aux[df_recomendacao_aux.ratio_recomenda < 1].copy()
    Freqencia.sort_values(['perc'], ascending= [0] ,inplace=True)
        
    resultado = RatioCorrigido.append(Freqencia)
    resultado.reset_index(inplace= True, drop= True)
    resultado.reset_index(inplace= True)
    
    dict_lista = {'cd_ChaveCliente' : 'sum',}
    df_preenche_vazios = df_recomendacao.reset_index().groupby('ultimo_produto').agg(dict_lista)
    df_preenche_vazios.sort_values(['cd_ChaveCliente'], ascending= [0] ,inplace=True)
    df_preenche_vazios.reset_index(inplace= True)
    lista1 = list(df_preenche_vazios.ultimo_produto.unique())
    lista2 = list(resultado.ultimo_produto.unique())
    aux = list(set(lista1) - set(lista2))
    df_preenche_vazios = df_preenche_vazios[df_preenche_vazios.ultimo_produto.isin(aux)].copy()
    df_preenche_vazios['chave_recomendacao'] = chave
    
    resultado.drop('index', axis= 1, inplace= True)
    resultado = resultado.append(df_preenche_vazios)
    resultado.reset_index(inplace= True, drop= True)
    resultado.reset_index(inplace= True)
    
    resultado = resultado.pivot(index='chave_recomendacao', columns='index', values='ultimo_produto')
    return resultado


# In[6]:

def junta_recomendacao(df_base, nivel):
    dict_lista = {'cd_ChaveCliente' : 'count',}
    c = nivel #começa no 2 e vai ate 10
    df_recomendacao = df_base.reset_index().groupby(colunas[:c]).agg(dict_lista)
    df_recomendacao.sort_values(['cd_ChaveCliente'], ascending= [0] ,inplace=True)
    df_recomendacao.reset_index(inplace= True)
    
    df_recomendacao = cria_chave(df_recomendacao)
    df_recomendacao = df_recomendacao[df_recomendacao.ultimo_produto != 'fim']
    listaChaves = list(df_recomendacao.chave_recomendacao.unique())
    
    result = calcula_recomendacao(df_recomendacao, listaChaves[0])
    for i in range(1,len(listaChaves)):
        result_aux = calcula_recomendacao(df_recomendacao, listaChaves[i])
        result = pd.concat([result, result_aux])
    return result


# 
# 
# 

# In[7]:

df_MatrizRecomendacao = pd.read_pickle('/home/felipe/Algoritmos_todosProdutos/Recomendacao/df_MatrizRecomendacao_24102017.pkl')


# In[23]:

df_base = pd.read_csv('/home/felipe/Algoritmos_todosProdutos/Churn_Consumo_Recomendacao.csv'
                      , error_bad_lines = False
                      , sep=';'
                      , dtype= {7: str}
                      , encoding='latin-1'
                      , header = 0) 


# In[24]:

df_base['Status'] = ['ativo' if s in ['Ativo', 'Atendido', 'Em ativação',
                                      'Aguardando ativação'] else 'inativo'
                     for s in df_base.Status]
df_base['nr_PrecoMensal'] = [x.replace(',', '.') for x in df_base.nr_PrecoMensal]
df_base['nr_PrecoMensal'] = df_base.nr_PrecoMensal.astype(float)
df_base = df_base[(df_base.nr_PrecoMensal > 0) & (df_base.Status == 'ativo')].copy()
df_base.drop_duplicates(['Provisioning'], keep='last', inplace= True)


# In[25]:

col_datas = ['Data_Desativacao', 'Data_Fim', 'Instalacao', 'dt_Reativacao', 'Primeiro_Servico_LW']
converte_datetime(df_base, col_datas)
df_base.sort_values(['Instalacao'], ascending= 1 ,inplace=True)


# In[26]:

df_base = df_base[df_base.Servico != 'Descontinuados'].copy()
df_base = df_base[df_base.Servico != 'Parcerias'].copy()
df_base = df_base[df_base.Servico != 'One Drive'].copy()
df_base = df_base[df_base.Servico != 'STREAMING_AUDIO_VIDEO'].copy()
df_base = df_base[df_base.Servico != 'Orago'].copy()
df_base = df_base[df_base.Servico != 'WR'].copy()


# In[27]:

df_base = df_base.groupby('cd_ChaveCliente').apply(contagens_anteriores)


# In[28]:

df_base.drop_duplicates(['Provisioning'], keep='last', inplace= True)
df_base.sort_values(['Instalacao'], ascending= 1 ,inplace=True)
colunas = ['cd_ChaveCliente', 'Provisioning', 'Servico', 'contagens_anteriores', 'Instalacao']
df_base = df_base[colunas][df_base.contagens_anteriores <= 10].copy()
df_base['Servico'] = ['PABX Virtual' if s == 'PABX Virtual ' else s for s in df_base.Servico]


# In[47]:

data = df_base.Instalacao.max()


# In[30]:

dict_lista = {'contagens_anteriores' : 'max',}
df_aux_ContagensAnteriores = df_base.groupby('cd_ChaveCliente').agg(dict_lista)
df_aux_ContagensAnteriores = df_aux_ContagensAnteriores.contagens_anteriores + 2
df_base = df_base.pivot(index='cd_ChaveCliente', columns='contagens_anteriores', values='Servico').copy()
colunas = ['Produto1','Produto2','Produto3','Produto4','Produto5','Produto6','Produto7','Produto8','Produto9','Produto10','Produto11']
df_base.columns = colunas
df_base.sort_values(colunas, ascending= [1,1,1,1,1,1,1,1,1,1,1] ,inplace=True)
df_base = pd.concat([df_aux_ContagensAnteriores, df_base], axis=1)    
df_base.reset_index(inplace= True)
df_base.rename(columns= {'index': 'cd_ChaveCliente'}, inplace= True)
df_base.fillna('fim', inplace = True)
df_base['chave_recomendacao'] = df_base.Produto1+df_base.Produto2+df_base.Produto3+df_base.Produto4+df_base.Produto5+df_base.Produto6+df_base.Produto7+df_base.Produto8+df_base.Produto9+df_base.Produto10
df_base['chave_recomendacao'] = [chave.replace('fim','') for chave in df_base.chave_recomendacao]


# In[31]:

df_base = pd.merge(df_base, df_aux_ContagensAnteriores.reset_index(), how='inner', on=['cd_ChaveCliente'])


# In[32]:

Recomendacao = []
for i in range(len(df_MatrizRecomendacao.columns)):
    Recomendacao.append('Recomendacao'+str(i+1))
df_base = pd.merge(df_base, df_MatrizRecomendacao[Recomendacao].reset_index(), how='inner', on=['chave_recomendacao'])


# In[38]:

df_base.shape


# In[42]:

colunas = Recomendacao
colunas.append('chave_recomendacao')
colunas.append('cd_ChaveCliente')


# In[ ]:

NomeCSV = '/home/felipe/Algoritmos_todosProdutos/Recomendacao/Recomendacao_'+str(df_base.Instalacao.max().year)+'_'+str(df_base.Instalacao.max().month)+'_'+str(df_base.Instalacao.max().day)+'.csv'
df_base[colunas].to_csv(NomeCSV)


# #  

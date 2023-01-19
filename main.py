#uvicorn main:app

import os
import sys
from functions import recommender
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pyspark.sql import SparkSession

from tabulate import tabulate

os.environ['SPARK_HOME'] = 'C:\spark'
os.environ['HADOOP_HOME'] = 'C:\spark\hadoop'
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}   

@app.get("/montar_dados")
def montarSpark():
    global dados_projecao
    spark = SparkSession\
        .builder\
        .appName('Modelo_Recomendacao')\
        .getOrCreate()
    dados_projecao = spark.read.parquet('DADOS/dataset_projecao_kmeans')
   
    return {"dados montados"}

@app.get('/procura/{id_imovel}')
def montarProcura(id_imovel: str):
    recomendadas = recommender(id_imovel, dados_projecao)
    teste = recomendadas.limit(5).toPandas()
    tabela_html = tabulate(teste,headers='keys', tablefmt='html')
    return HTMLResponse(content=tabela_html)

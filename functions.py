from scipy.spatial.distance import euclidean
from pyspark.sql.types import FloatType
from pyspark.sql import functions as f


def calculate_euclidean_distance(imovel, valor):
    return euclidean(imovel, valor)


def recommender(id_imovel, dataframe_kmeans):
    cluster = dataframe_kmeans\
        .filter(dataframe_kmeans.id == id_imovel)\
        .select('cluster_pca')\
        .collect()[0][0]

    imoveis_recomendadas = dataframe_kmeans\
        .filter(dataframe_kmeans.cluster_pca == cluster)

    imovel_procurado = imoveis_recomendadas\
        .filter(imoveis_recomendadas.id == id_imovel)\
        .select('pca_features')\
        .collect()[0][0]

    euclidean_udf = f.udf(lambda x: calculate_euclidean_distance(
        imovel_procurado, x), FloatType())

    colunas_nao_utilizadas = [
        'features', 'scaled_features', 'pca_features', 'cluster_pca', 'distancia']

    recomendadas = imoveis_recomendadas\
        .withColumn('distancia', euclidean_udf('pca_features'))\
        .select([col for col in imoveis_recomendadas.columns if col not in colunas_nao_utilizadas])\
        .orderBy('distancia')

    return recomendadas

# Filtragem bidimensional de uma imagem

Filtragem de uma imagem utilizando algum dos filtros abaixo. Após o filtro a imagem será comparada com um dataset para classifica-la segundo o algoritmo knn.

Para usar o programa deve-se entrar com: <br/>
-Nome da imagem a ser filtrada <br/>
-Filtro (1 - arbitrário, 2 - laplaciana da gaussiana, 3 - operador sobel) <br/>
-Parâmetros do filtro <br/>
-Posições de corte (Hlb, Hub, Wlb, Wub) <br/>
-Nome do arquivo .npy para dataset <br/>
-Nome do arquivo .npy para labels <br/>

Exemplos de entrada:

cat.png <br/>
1 <br/>
3 3 <br/>
7.4 5.2 2.1 <br/>
0.2 0.2 0.1 <br/>
0.3 0.9 6.1 <br/>
0.1 0.2 0.3 0.4 <br/>
mnist.npy <br/>
mnist_labels.npy <br/>

cat.png <br/>
2 <br/>
5 <br/>
1.7 <br/>
0.1 0.2 0.3 0.4 <br/>
mnist.npy <br/>
mnist_labels.npy <br/>

cat.png <br/>
3 <br/>
0.1 0.2 0.3 0.4 <br/>
mnist.npy <br/>
mnist_labels.npy <br/>

library(data.table) #pacote para importar base
library(tidyverse)  #pacote para manipular dados
library(janitor)    #pacote para padronizar nome das colunas
library(DescTools)  #
library(corrplot)   #cria "gráfico" matriz correlação
library(e1071)      #pacote com função para montar modelo Svm
library(caTools)    #
library(clusterGeneration) #pacote para separar base em teste e treino
library(caret)             #pacote com algoritmos de machine learnign
library(knitr)  

#importando base:
df = fread("adult.csv",stringsAsFactors = T,header = T)

setnames(df,"income","target")

names(df) = str_replace(names(df),"-","_")

#verificando se existem dados faltantes:
df %>% 
  summarise_all(funs(sum(is.na(.))))

df$native_country = NULL

#verificando levels por fator:
df %>%
  select_if(is.factor) %>% 
  sapply(levels)

#reorganizando variável workclass:
df$workclass <- as.character(df$workclass)

df$workclass[df$workclass == "Without-pay" | 
               df$workclass == "Never-worked"] = "Unemployed"

df$workclass[df$workclass == "State-gov" |
               df$workclass == "Local-gov"] = "SL-gov"

df$workclass[df$workclass == "Self-emp-inc" |
               df$workclass == "Self-emp-not-inc"] = "Self-employed"

#voltando variável para fator:
df$workclass = as_factor(df$workclass)

#convertendo variável para char:
df$marital_status = as.character(df$marital_status)
#reorganizando níveis da variável marital status:
df$marital_status[df$marital_status == "Divorced" | 
                    df$marital_status == "Separated"|
                    df$marital_status == "Widowed"  |
                    df$marital_status == "Never-married"] = "Not_Married"

df$marital_status[df$marital_status == "Married-civ-spouse" |
                    df$marital_status == "Married-spouse-absent"|
                    df$marital_status == "Married-AF-spouse"] = "Married"

#voltando variável para fator
df$marital_status = as_factor(df$marital_status)

#criando vetor com nome das colunas que são categóricas:
nomes = df %>%
  select_if(is.factor) %>% 
  names()

#lista para armazenar gráficos:
lista_grafico = list()


#loop para criar gráfico de barras para as variáveis categóricas:
for (i in 1:length(nomes)){ 
  
  lista_grafico[[i]] = ggplot(df,aes(fct_infreq(!!sym(nomes[i])),fill = !!sym(nomes[i])))+
    geom_bar(stat = "count",show.legend = F)+
    xlab(nomes[i]) + ylab("Quantidade")+ggtitle(paste("Grafico",nomes[i],sep = " "))+
    theme_bw()+
    geom_text(stat = 'count', 
              aes(label = paste0(100*round(stat(count)/nrow(df),digits = 3),"%"),
                  vjust = -0.2))+
    theme(axis.text.x = element_text(angle = 90, vjust = 1))
}

#matriz de correlação:
df_correlacao = df %>% select_if(is.numeric) %>% cor()
corrplot.mixed(df_correlacao,lower.col = "black", number.cex = .7)

#cria lista para armazenar gráficos de barra separados de acordo com a variável resposta:
lista_segmenta = list()
#grafico de barras separado pela variável resposta: 
for(i in 1:length(nomes)){
  
  lista_segmenta[[i]] = ggplot(df,aes(!!sym(nomes[i]),fill = target))+
    geom_bar(stat = "count",position = position_dodge())+
    facet_grid(~target)+
    theme_bw()+
    xlab(nomes[i])+
    ggtitle(nomes[i])+
    theme(axis.text.x = element_text(angle = 90, vjust = 1),
          plot.title = element_text(hjust = 0.5))+
    geom_text(stat = 'count',aes(label = paste0(100*round(stat(count)/nrow(df),
                                                          digits = 3),"%"),vjust = -0.2))
  
  
}

for(i in 1:(length(nomes)-1)){
  print(lista_segmenta[[i]])
}

#histograma com horas de trabalho por semana:
ggplot(df,aes(x = df$`hours_per-week`,fill=df$target))+geom_histogram(color = "black",binwidth = 2)+
  theme_bw()+
  xlab("Hours Per Week")

#gráfico da densidade da idade:
ggplot(df,aes(x = age, fill = target))+geom_histogram(binwidth = 2,colour="black")+theme_bw()+
  ggtitle("Histogram - Age")+theme(plot.title = element_text(hjust = 0.5))


#medidas resumo das variáveis numéricas:
df %>% select_if(is.numeric) %>% summary() %>% kable()

#teste qui quadrado, para verificar se alguma variável é individualmente independente 
#da variável resposta. Caso seja,essa variável não entraria no modelo
df_fator = df %>% select_if(is.factor)


#armazenando p-valor, porém algumas aproximações não foram corretas
#(provavelmente pelo fato de ter menos de cinco em alguma casela da matriz de contagem) 
#nesse caso o teste exato de fisher é conveniente:
x = df %>% 
  select_if(is.factor) %>%
  summarise_all(funs(chisq.test(.,df$target)$p.value))

#separando a base de treino e teste:
intrain = createDataPartition(y = df$target,p = 0.75,list = F)
treino = df[intrain,]
teste = df[-intrain,]

#criando modelo svm:
modelo_svm = svm(treino$target ~ .,data = treino)

#criando predição:
variavel_resposta = teste %>% dplyr::select(target)

#cria base sem a variável resposta:
teste = teste %>% dplyr::select(-target)

#realiza a predicao usando o modelo svm:
predicao_svm = predict(modelo_svm,teste)

#criando matriz de confusão para o modelo svm:
confusionMatrix(variavel_resposta$target,predicao_svm)$table
)


#criando modelo de arvore de decisão:
modelo_arvore = rpart::rpart(treino$target ~ .,data = treino)

#predicao modelo arvore:
predicao_arvore = predict(modelo_arvore,teste,type = "class")

#criando matriz de confusão:
confusionMatrix(variavel_resposta$target,predicao_arvore)

#variaveis importantes na árvore:
importancia_variavel = varImp(modelo_arvore)

#criando matriz de confusão:
confusionMatrix(variavel_resposta$target,predicao_arvore)

x = add_rownames(as_data_frame(modelo_arvore$variable.importance))

#cria gráfico com importancia das variaveis
ggplot(x,aes(reorder(x$rowname,x$value),x$value))+
  geom_bar(stat = "identity",color = "black",fill = "black")+
  coord_flip()+
  xlab("Variable")+
  ylab("Value")+
  theme_bw()+
  ggtitle("Variable Importance")

#modelo de regressão logistica com variáveis de importancia:
treino_2 = treino %>% dplyr::select(relationship,marital_status,capital_gain,target)
modelo_logistico = glm(target ~ .,treino_2,family = "binomial")


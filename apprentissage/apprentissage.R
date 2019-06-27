library(bnlearn)
#library(Rgraphviz)

args = commandArgs(trailingOnly=TRUE)
if(length(args) != 3){
    stop("Le script attend le nom du fichier de données en entrée, le nom du
          fichier en sortie pour les arcs et le nom du fichier en sortie
          pour les paramètres appris",
         call.=FALSE)
}

input_file_name = args[1]
arcs_file_name = args[2]
parameters_file_name = args[3]

donnees <- read.csv(file=input_file_name, header=TRUE, sep=",", row.names="Date")
cat("Le fichier", input_file_name, "a été correctement chargé.\n")
#donnees <- donnees[,c("T_ext", "P_res")]
#head(donnees)

#donnees <- donnees[sample(nrow(donnees)),]
#size <- floor(0.8 * nrow(donnees))
#index <- sample(seq_len(nrow(donnees)), size = size)
#train_set  <- donnees[index, ]
#test_set  <- donnees[-index, ]

# Liste des arcs qu'on interdit
from = c("P_res_tp1","T_ext_tp1", "D_res_tp1", "T_depres_tp1", "T_retres_tp1")
to = c("P_res_t","T_ext_t", "D_res_t", "T_depres_t", "T_retres_t")
#from = c("T_ext_tp1")
#to = c("P_res_t","T_ext_t", "D_res_t", "T_depres_t", "T_retres_t")
bl = expand.grid(from, to)

print("Apprentissage de la structure...")
bn = hc(donnees, blacklist=bl, score="bge", restart=20, debug=FALSE)
print(bn)
#bn = hc(donnees, score="bge", restart=20)
#graphviz.plot(bn)

#bn = bn.cv(donnees, "hc", algorithm.args=list(score="bge"))
print("Apprentissage des paramètres...")
fit = bn.fit(bn, donnees)
print(fit)

#pred <- function(x) coefficients(fit$Puissance.réseau..kW.)[2] * (x - fit$Temp.Exterieure...C.$sd) + coefficients(fit$Puissance.réseau..kW.)[1]

#Y_pred = apply(test_set[1], 2, pred)
#diff = Y_pred - test_set[2]
#s1 = apply(diff, 2, sd)
#diff = apply(diff, 2, abs)

#m = colMeans(diff)
#s = apply(diff, 2, sd)

#print("Fit")

#print(m)
#print(s)
#print(s1)

# Ecriture des arcs
write.table(bn$arcs,
            file=arcs_file_name,
            quote=FALSE,
            sep="->",
            row.names=FALSE,
            col.names=FALSE,
            eol="\n",
            append=FALSE)

cat("{", file=parameters_file_name)
for (i in 1:ncol(donnees)){
    write.table(fit[[i]][[1]],
                file=parameters_file_name,
                row.names=FALSE,
                quote=TRUE,
                col.names=FALSE,
                eol="",
                append=TRUE)
    cat(":{", file=parameters_file_name, append=TRUE)
    write.table(paste("\"SD\"",fit[[i]][[5]]),
                file=parameters_file_name,
                row.names=FALSE,
                quote=FALSE,
                col.names=FALSE,
                eol="\n",
                append=TRUE)
    write.table(fit[[i]][[4]],
                file=parameters_file_name,
                row.names=TRUE,
                quote=TRUE,
                col.names=FALSE,
                append=TRUE)
    cat("},", file=parameters_file_name, append=TRUE)
}
cat("}", file=parameters_file_name, append=TRUE)

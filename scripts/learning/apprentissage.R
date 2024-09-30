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

donnees <- read.csv(file=input_file_name, header=TRUE, sep=",")
cat("Le fichier", input_file_name, "a été correctement chargé.\n")

print("Apprentissage de la structure...")
bn = hc(donnees, score="bge", restart=20, debug=FALSE)
print(bn)
#bn = hc(donnees, score="bge", restart=20)
#graphviz.plot(bn)

print("Apprentissage des paramètres...")
fit = bn.fit(bn, donnees)
print(fit)

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

setwd("C:/Users/Lucas/Cursos/PucRio/Metaheuristicas/Projeto1/Projeto1_LOP/Projeto1_LOP/ls analysis 2")
data = read.csv("report.txt", sep = ' ', comment.char = '!')
ndi_combinations = unique(cbind(data$n,data$d,data$i))

ls_names = unique(data$ls_name)
strg_names = unique(data$move_strg)
alg_combinations = rbind(
  c("tree","first_move"),
  c("tree","best_move")
)

ret = matrix(nrow = nrow(ndi_combinations),ncol = nrow(alg_combinations));
rownames(ret) = rep("",nrow(ret));
colnames(ret) = rep("",ncol(ret));

#foreach ndi comb
  #foreach alg_comb
    #get data for ndi-alg
for (i in 1:nrow(ndi_combinations))
{
  nn = ndi_combinations[i,1]; dd = ndi_combinations[i,2]; ii = ndi_combinations[i,3];
  rownames(ret)[i] = toString(ndi_combinations[i,])
  
  for(j in 1:nrow(alg_combinations)){
    #print(ndi_combinations[i,]); print(alg_combinations[j,]);
    ls_name_ = alg_combinations[j,1]; mv_name_ = alg_combinations[j,2];
    #select data from ndi-alg
    selected_dt = subset(data , n == nn & d == dd & i == ii &
                           ls_name == ls_name_ & move_strg == mv_name_ 
                           ,select = c(fo_mean,fo_sd) 
                         ); 
    
    cv_ = selected_dt$fo_sd/selected_dt$fo_mean;
    
    ret[i,j] = paste(selected_dt$fo_mean, " (", round(cv_,5), ")");
    
    colnames(ret)[j] = toString(alg_combinations[j,]);
  }
}

print("Objective Function Mean after 85 sec repeatedly local searching")
print(ret)

print("Tree-first fo_mean  > Tree-best?");
print(table(ret[,1] > ret[,2]))
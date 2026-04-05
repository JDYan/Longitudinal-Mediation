clear;
clc;
% import common list
gene_list=importdata('...\Brain_environment_association\Control_association\Gene_covariate\new_list.mat');
old_list=importdata('...\Brain_environment_association\Control_association\early_development\time_all\New_data\new_list.mat');
% import old mat
gene=importdata('...\Brain_environment_association\Control_association\Gene_covariate\new_mat.mat');
old_input=importdata('...\Brain_environment_association\Control_association\early_development\time_all\New_data\input_data.mat');
old_family=importdata('...\Brain_environment_association\Control_association\early_development\time_all\New_data\family.mat');
old_mat=importdata('...\Brain_environment_association\Control_association\early_development\time_all\New_data\new_mat.mat');
% create new mat
new_input=zeros(size(old_input));
new_family=old_family;
new_mat=zeros(size(old_mat));
new_gene=zeros(size(gene));
new_list=gene_list;
% start finding common
count=0;
for i=1:size(gene_list,1)
    temp=gene_list(i,:); % need to change for different situation
    [L,loc]=ismember(temp,old_list,'rows');
    if L
        count=count+1;
        new_input(count,:)=old_input(loc,:);
        new_family(count,:)=old_family(loc,:);
        new_list(count,:)=gene_list(i,:);
        new_gene(count,:)=gene(i,:);
        new_mat(count,:)=old_mat(loc,:);
    end
end
input_data=new_input(1:count,:);
input_data=zscore(input_data);
family=new_family(1:count,:);
new_list=new_list(1:count,:);
new_mat=new_mat(1:count,:);
new_mat=zscore(new_mat);
new_gene=new_gene(1:count,:);
new_gene=zscore(new_gene);
% control new_mat with new_gene
temp=new_mat;
temp_result=zeros(size(temp));
for i=1:size(temp,2)
    x=temp(:,i);
    Xc=new_gene;
    model=fitlm(Xc,x);
    coefficients=model.Coefficients.Estimate;
    intercept=coefficients(1);
    y_coeff=coefficients(2:end);
    x_new=x-Xc*y_coeff-intercept;
    temp_result(:,i)=x_new;
end
new_mat=zscore(temp_result);
% save
save_path='...\Brain_environment_association\Control_association\early_development\time_all\New_data\input_data.mat';
save(save_path,'input_data');
save_path='...\Brain_environment_association\Control_association\early_development\time_all\New_data\family.mat';
save(save_path,'family');
save_path='...\Brain_environment_association\Control_association\early_development\time_all\New_data\new_list.mat';
save(save_path,'new_list');
save_path='...\Brain_environment_association\Control_association\early_development\time_all\New_data\new_mat.mat';
save(save_path,'new_mat');

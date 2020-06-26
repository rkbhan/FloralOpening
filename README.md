---
title: "Machine learning and Bayesian Inference for High Throughput Phenotyping of Floral Opening Time"
author: "Rongkui Han"
date: "6/26/2020"
output: 
  html_document: 
    keep_md: yes
---
## Machine learning and Bayesian Inference for High Throughput Phenotyping of Floral Opening Time

### Introduction

Flower opening and closure are traits of reproductive importance in all angiosperms, because they determine the success of self- and cross-pollination events. Existing variations in floral opening hours have been recorded in many species, but the transient nature of this phenotype has rendered it a difficult target for genetic studies. In this repository, I describe a two-step method that infers peak floral opening time of different plant accessions from time-stamped drone images.

### Step 1

In FOH_Machine_Learning.md, you will find a simple method using support vector machine (SVM) to identify flowers from image series obtained by a drone-mediated remote sensing phenotyping experiment.

### Step 2

In FOH_Bayesian_Inference.md, you will find a Bayesian method that finds the best fitted Gaussian-like curve for each plot's unique floral pixel count ptofile.

All datasets used here can be found under FOH_GitHub_files. I hope you find this pipeline helpful!


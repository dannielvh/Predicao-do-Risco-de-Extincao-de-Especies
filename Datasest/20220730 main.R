#' General predictors of extinction risk
#' Author: @XXXXXXX
#' Date: 2022-05-27
#' 
#' 

#### 0. INTRO ####

#' This script has four parts:
#' 1. Functions
#' 2. Data preparation
#'   .1 Import/Filter/Structure data 
#'     - Dataset
#'     - log_rules
#'     - bean_plots
#'     - data resolution
#'     - names of traits
#'     - random terms used
#'   .2 Derived variables
#'     - threatened/non-threatened binary response
#'     - altitudinal range
#'     - relative altitudinal range
#'     - relative offspring size
#'   .3 Log-transform traits
#'     - based on log_rules dataset
#'     
#' 3. Analyses
#'   - GLMMs
#'   - meta-analyses
#'   
#' 4. Presenting results
#'   - Number of sp. per trait*taxa
#'   - % imputed data per trait*taxa
#'   - forest-plots
#'   
#'   

# Whether to run bootstrap analyses or get them from disk, if False, need to have 
# them on disk
run_bootstraps = T

#### 1. Functions ####
 
#' Runs meta-analyses for each trait (fixed effects) using package metafor.
#' Uses as arguments the estimate and confidence intervals of the glmm output of 
#' function run_my_glmms()
run_my_meta_analyses <- function(glmm_outputs){
  library(metafor)
  meta_objects <- list() # To store metafor outputs
  
  for (i in seq_along(glmm_outputs)){
    test <- glmm_outputs[[i]]
    if(length(na.exclude(test$Slope)) > 0){
      meta_objects[[i]] <- rma(yi  = test$Slope,
                               sei = test$Std_err, 
                               method = "FE", slab = test$Taxa
      )
    } else {
      meta_objects[[i]] <- NA
    }
  }
  names(meta_objects) <- names(glmm_outputs)
  return(meta_objects)
}

#' Log transform
#' @param x list of dataframes of species x traits (all)
#' @param y matrix that says on which traits per group log transformation should be done
#' same as x, log transformed where it should be log transformed
log_transform <- function(x, y){
  for (i in 1:length(x)){
    group <- names(x)[i]
    for (j in 1:length(y[,1])){
      if (y[j,i+1, drop = T]){
        x[[group]][,as.character(y[j,1, drop = T])] <- 
          log(x[[group]][,as.character(y[j,1, drop = T])])
      }
    }
  }
  return(x)
}

### Clean log_rules file and bean_plots files
quick_clean<-function(x){
  log_rules <- as.data.frame(x)
  names(log_rules)[1] <- "Trait"
  log_rules[is.na(log_rules)] <- FALSE
  log_rules <- as.data.frame(apply(log_rules, 2, function(x) unlist(as.character(x))))
  log_rules[2:ncol(log_rules)] <- apply(log_rules[2:ncol(log_rules)],2, as.logical)
  return(log_rules)
  
}

#' Extracts estimate, lower, and upper confidence intervals
#' @param x: "rma.uni" object 
#' @returns vector of length 3 with estimate, lower and upper conf int
extract_meta_results <- function(x){
  estimate <- NA
  ci.lb    <- NA
  ci.ub    <- NA
  
  if (class(x)[1] == "rma.uni"){
    estimate <- as.numeric(x$b)
    ci.lb    <- x$ci.lb
    ci.ub    <- x$ci.ub
  }
  return(c(estimate,ci.lb,ci.ub))
}

# First: extract slope parameters and confidence intervals from each glmmTMB
# object:
glmmTMB_to_Dataframe <- function(glmm_results){
  trait_taxa <- vector("list", length = length(glmm_results))
  names(trait_taxa) <- names(glmm_results)
  for (i in seq_along(glmm_results)){
    M0_summary <- data.frame(Taxa = names(glmm_results[[i]]), 
                             Slope          = NA,
                             Slope_2.5      = NA,
                             Slope_97.5     = NA,
                             Std_err        = NA,
                             Df             = NA,
                             Observations   = NA,
                             Convergence    = NA)
    
    for (j in seq_along(glmm_results[[i]])){
      
      M0 <- glmm_results[[i]][[j]]
      
      if (class(M0) == "glmmTMB") { 
        M0_summary[j,"Convergence"]    <- M0$fit$convergence
        M0_summary[j,"Slope"]        <- confint(M0)[2,3]
        M0_summary[j,"Slope_2.5"]    <- confint(M0)[2,1]
        M0_summary[j,"Slope_97.5"]   <- confint(M0)[2,2]
        M0_summary[j,"Std_err"]      <- coefficients(summary(M0))$cond[2,2]
        M0_summary[j,"Observations"] <- nobs(M0)
        M0_summary[j, "Df"]          <- df.residual(M0)
      }
    }
    trait_taxa[[i]] <- M0_summary
  }
  return(trait_taxa)
}


#' Function that plots the effect size figure per trait*taxa, plus the meta-
#' analysis results
plot_nice_figure <- function(glmm_summary, meta_summary, trait_labs, file_name = "effect sizes", 
                             width = 19, height = 15, dpi = NULL, device = "pdf"){
  longdata <- NULL
  has.meta_summary <- !is.null(meta_summary)
  
  names(glmm_summary) <- trait_labs
  
  if(has.meta_summary){
    for (i in seq_along(meta_summary)){
      names(meta_summary[[i]]) <- trait_labs
    }
  }
  
  for ( i in seq_along(glmm_summary)){
    Trait   <- names(glmm_summary)[i]
    this    <- glmm_summary[[i]][,1:4]  # Select only relevant columns
    # Rbind meta-analysis results
    if(has.meta_summary){
      Vertebrates   <- extract_meta_results(meta_summary[[1]][[i]])
      Invertebrates <- extract_meta_results(meta_summary[[2]][[i]])
      Plants        <- extract_meta_results(meta_summary[[3]][[i]])
      this <- rbind(this[1:5,], c("", Vertebrates),
                    this[6:10,],c("  ", Invertebrates),
                    this[11:15,], c("   ", Plants))
      Type  <- rep(c(rep("GLMM",5), "Meta-analysis"),3) 
      Group <- c(rep("Vertebrate",6), rep("Invertebrate",6),rep("Plant",6))
      current <- cbind(Group,Type,Trait,this)
    }else
      current <- cbind(Trait,this)
    
    longdata <- rbind(longdata,current)
  }
  longdata
  #Make numeric columns numeric
  
  if(has.meta_summary)
    cols.num <- c(5:7)
  else
    cols.num <- c(3:5)
    
  longdata[cols.num] <-sapply(longdata[cols.num],as.numeric)
  str(longdata)
  
  # Add column: whether it is significant or not
  signif <- rep("", nrow(longdata))
  for (i in seq_along(longdata[,1])){
    if (!any(is.na(longdata[i,cols.num])) && (all(longdata[i,cols.num] > 0) || all(longdata[i,cols.num] < 0)))
      signif[i] <- "*"
  }
  longdata$signif <- signif
  
  # Trait and Taxa columns should be ordinal
  longdata$Trait <- factor(longdata$Trait, ordered = T, levels = 
                             unique(longdata$Trait))
  #longdata$Taxa <- factor(longdata$Taxa, ordered = T, levels = 
  #                            rev(unique(longdata$Taxa)))
  longdata$Taxa <- factor(longdata$Taxa, ordered = T, levels = 
                            unique(longdata$Taxa))
  if(has.meta_summary){
    longdata$Group <- factor(longdata$Group, ordered = T, levels = 
                               unique(longdata$Group))
  }
  
  # Inspiration from:
  # https://datascienceplus.com/lattice-like-forest-plot-using-ggplot2-in-r/
  # Extracted on 9th of May, 2022
  
  if(has.meta_summary){
  p = ggplot(data=longdata,
             aes(x = Taxa,y = Slope, ymin = Slope_2.5,
                 ymax = Slope_97.5, size = Type,label = signif))+
    geom_pointrange(aes(col=Group, shape = Type))+
    geom_hline(aes(fill=Taxa),yintercept =0, linetype=2)+
    geom_vline(aes(fill=Taxa),xintercept =c(6.5, 12.5), linetype=1, colour = "gray")+
    scale_size_manual(values = c(0.3,0.8))+
    scale_shape_manual(values = c(16,18))+
    scale_color_manual(values = c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"))+
    scale_y_continuous(position = "right", expand = expansion(mult = c(0.1, 0.1)))+
    xlab('Taxa')+ ylab("Log-odds of extinction threat (95% Confidence Interval)")+
    #geom_errorbar(aes(ymin=Slope_2.5, ymax=Slope_97.5,col=Group),width=0.5,cex=1)+
    geom_errorbar(aes(ymin=Slope_2.5, ymax=Slope_97.5,col=Group, size = Type))+#
    geom_text(aes(y = stat(ymax)), label = signif, size = 5)+
    facet_wrap(~Trait,strip.position="left",nrow=5,scales = "free_y") +
    #facet_wrap(~Trait,strip.position="left",nrow=9,scales = "fixed") +
    theme(
          panel.background = element_blank(), 
          plot.title=element_text(size=16,face="bold"),
          axis.ticks.x=element_blank(),
          axis.text.x=element_text(face="bold",angle=90, hjust=1, vjust = 0.4),
          axis.title.x=element_text(size=12,face="bold",angle=0),
          axis.title.y.right=element_text(size=12,face="bold",angle=90),
          strip.text.y = element_text(angle=90,face="bold", size = 7,
                                      margin = margin(0.2,0.1,0.2,0.1,"cm")),
          legend.position = "none",
          #axis.text.y=element_text(face="bold",angle=0),
          #)+
    )
  #coord_flip()
  }else {
    p = ggplot(data=longdata,
               aes(x = Taxa,y = Slope, ymin = Slope_2.5,
                   ymax = Slope_97.5, label = signif))+
      geom_pointrange(col="#E69F00")+
      geom_hline(aes(fill=Taxa),yintercept =0, linetype=2)+
      #geom_vline(aes(fill=Taxa),xintercept =c(6.5, 12.5), linetype=1)+
      scale_size_manual(values = c(0.3,0.8))+
      scale_shape_manual(values = c(16,18))+
      scale_color_manual(values = c("#E69F00"))+
      scale_y_continuous(position = "right", expand = expansion(mult = c(0.1, 0.1)))+
      xlab('Taxa')+ ylab("Log-odds of extinction probability (95% Confidence Interval)")+
      #geom_errorbar(aes(ymin=Slope_2.5, ymax=Slope_97.5,col=Group),width=0.5,cex=1)+
      geom_errorbar(aes(ymin=Slope_2.5, ymax=Slope_97.5), col="#E69F00")+#
      geom_text(aes(y = stat(ymax)), label = signif, size = 5)+
      #facet_wrap(~Trait,strip.position="left",nrow=9,scales = "free_y") +
      facet_wrap(~Trait,strip.position="left",nrow=5,scales = "fixed") +
      theme(panel.background = element_blank(), 
            plot.title=element_text(size=16,face="bold"),
            axis.ticks.x=element_blank(),
            axis.text.x=element_text(face="bold",angle=90, hjust=1, vjust = 0.4),
            axis.title.x=element_text(size=12,face="bold",angle=0),
            axis.title.y.right=element_text(size=12,face="bold",angle=90),
            strip.text.y = element_text(angle=90,face="bold", size = 8),
            legend.position = "none",
            #axis.text.y=element_text(face="bold",angle=0),
            #)+
      )
    
  }
  
  if(file_name == "")
    return(p)
  else
    ggsave(file_name,
           device = device, units = "cm", width = width, height = height, dpi = dpi )
}

#' Runs glmmTMB models
#' @param type string, one of three implemented 
#' options: "linear", "quadratic", or "body size interaction"
#' @param dt list of dataframes similar to all_log
#' @param traits vector of strings with all the traits to use
#' @param random_terms vector of strings with random terms to use for each trait
#' @param min.obs.level minimum observations for each level of the 
#' response variable to run model
#' @param resolution list of dataframe similar to all_log but instead of trait values 
#' it contains metadata (resolution)
#' @returns a list of fitted glmmTMB models of size length(traits)
run_my_glmms <- function(dt, traits, random_terms, min.obs.level){
  library(glmmTMB)
  trait_taxa <- vector("list", length(traits))
  names(trait_taxa) <- traits
  
  for (i in seq_along(traits)){
    taxa <- vector("list", length(dt))
    names(taxa) <- names(dt)
    for (j in seq_along(dt)){
      run_model = T
      
      # Isolate trait
      dt[[j]]$my_trait <- dt[[j]][,traits[i]]
      
      # sample size for each binary outcome threatened/non-threatened?
      sp.size <- table(dt[[j]]$StatusC[which(!is.na(dt[[j]]$my_trait))])
      nonThreatObs  <- ifelse(is.na(sp.size["0"]), 0 , sp.size["0"])
      ThreatObs     <- ifelse(is.na(sp.size["1"]), 0 , sp.size["1"])
      
      if (any(c(nonThreatObs,
                ThreatObs) < min.obs.level)){ 
        #Only continue with the model part if sample size
        # is at least 3 for each level of the binary response
        run_model = F
      }
      
      # If my_trait is binary, run model only if all combinations of
      # my_trait and threat status have observations to avoid huge
      # estimate variances
      if (length(unique(na.exclude(dt[[j]]$my_trait))) == 2){
        comb.numbers <- 
          table(dt[[j]][which(!is.na(dt[[j]]$my_trait)),c("my_trait", "StatusC")])
        if (any(comb.numbers == 0))
          run_model = F
      }
      
      if (run_model){
        # report where we are
        print(paste(traits[i], names(taxa)[j]))
        # run the model
        tryCatch({
          M0 <- glmmTMB(as.formula(
            paste0("StatusC ~ scale(my_trait) ",random_terms[j])),
            data = dt[[j]], family = "binomial")
          taxa[[j]] <- M0
        },
        error=function(cond) {
          message("Here's the original error message:")
          message(cond)
          # Choose a return value in case of error
          M0 <- NA
        },
        warning=function(cond) {
          message("Here's the original warning message:")
          message(cond)
          M0 <- glmmTMB(as.formula(
            paste0("StatusC ~ scale(my_trait) ",random_terms[j])),
            data = dt[[j]], family = "binomial")
          taxa[[j]] <- M0
          # Choose a return value in case of warning
        })
      }
    }
    trait_taxa[[i]] <- taxa
  }
  names(trait_taxa)[which(names(trait_taxa) %in% "RELATIVE_OFFSPRING_SIZE")] <- "OFFSPRING_SIZE" 
  names(trait_taxa)[which(names(trait_taxa) %in% "RELATIVE_ALTITUDINAL_RANGE")] <- "ALTITUDINAL_RANGE"
  return(trait_taxa)
  #return(M0)
}

#' Runs glmmTMB models
#' @param type string, one of three implemented 
#' options: "linear", "quadratic", or "body size interaction"
#' @param dt list of dataframes similar to all_log
#' @param traits vector of strings with all the traits to use
#' @param random_terms vector of strings with random terms to use for each trait
#' @param min.obs.level minimum observations for each level of the 
#' response variable to run model
#' @param resolution list of dataframe similar to all_log but instead of trait values 
#' it contains metadata (resolution)
#' @returns a list of fitted glmmTMB models of size length(traits)
run_my_glmms_bootstrap_df <- function(dt, traits, random_terms, min.obs.level, reps = 100){
  library(glmmTMB)
  trait_taxa <- vector("list", length(traits))
  names(trait_taxa) <- traits
  
  for (i in seq_along(traits)){
    taxa <- vector("list", length(dt))
    names(taxa) <- names(dt)
    for (j in seq_along(dt)){
      run_model = T
      
      # Isolate trait
      dt[[j]]$my_trait <- dt[[j]][,traits[i]]
      
      # sample size for each binary outcome threatened/non-threatened?
      sp.size <- table(dt[[j]]$StatusC[which(!is.na(dt[[j]]$my_trait))])
      nonThreatObs  <- ifelse(is.na(sp.size["0"]), 0 , sp.size["0"])
      ThreatObs     <- ifelse(is.na(sp.size["1"]), 0 , sp.size["1"])
      
      if (any(c(nonThreatObs,
                ThreatObs) < min.obs.level)){ 
        #Only continue with the model part if sample size
        # is at least 3 for each level of the binary response
        run_model = F
      }
      
      # If my_trait is binary, run model only if all combinations of
      # my_trait and threat status have observations to avoid huge
      # estimate variances
      if (length(unique(na.exclude(dt[[j]]$my_trait))) == 2){
        comb.numbers <- 
          table(dt[[j]][which(!is.na(dt[[j]]$my_trait)),c("my_trait", "StatusC")])
        if (any(comb.numbers == 0))
          run_model = F
      }
      # If model runs but does not converge with original data, exclude it
      tryCatch({
        M0 <- glmmTMB(as.formula(
          paste0("StatusC ~ scale(my_trait) ",random_terms[j])),
          data = dt[[j]], family = "binomial")
      },
      error=function(cond) {
        message("Here's the original error message:")
        message(cond)
        # Choose a return value in case of error
        run_model = F
      },
      warning=function(cond) {
        message("Here's the original warning message:")
        message(cond)
        run_model = F
      })
      if (run_model){
        #bootstrap
        dts <- data.frame(Sample = 1:reps, 
                          Slope          = NA,
                          Slope_2.5      = NA,
                          Slope_97.5     = NA,
                          Std.Error      = NA)
        non_NAs       <- dt[[j]][which(!is.na(dt[[j]]$my_trait)),]
        threatened    <- non_NAs[which(non_NAs$StatusC == "1"),]
        nonthreatened <- non_NAs[which(non_NAs$StatusC == "0"),]
        # report where we are
        print(paste(traits[i], names(taxa)[j]))
        
        k <- 1
        while (k <= reps){
          th <- threatened[sample(1:nrow(threatened), 30, replace = T),]
          nt <- nonthreatened[sample(1:nrow(nonthreatened), 30, replace = T),]
          current_dt <- rbind(th, nt)
          # run the model
          print(k)
          #dts[k,2:4] <- tryCatch({
          tryCatch({
            M0 <- glmmTMB(as.formula(
              paste0("StatusC ~ scale(my_trait) ",random_terms[j])),
              data = current_dt, family = "binomial")
            dts[k,2:5] <- extract_effect_sizes(M0)
            k <- k + 1
          },
          error=function(cond) {
            message("Here's the original error message:")
            message(cond)
            # Choose a return value in case of error
            #dts[[k]] <- as.character(cond)
          },
          warning=function(cond) {
            message("Here's the original warning message:")
            message(cond)
            # Choose a return value in case of warning
            #dts[[k]] <- as.character(cond)
          })
          
          #and whether it converged
          #print(paste(M0$fit$convergence))
        }
        taxa[[j]] <- dts
        
      }
    }
    trait_taxa[[i]] <- taxa
  }
  return(trait_taxa)
  #return(M0)
}

# First: extract slope parameters and confidence intervals from each glmmTMB
# object:
glmmTMB_to_Dataframe2 <- function(glmm_results){
  trait_taxa <- vector("list", length = length(glmm_results))
  names(trait_taxa) <- names(glmm_results)
  for (i in seq_along(glmm_results)){
    M0_summary <- data.frame(Taxa = names(glmm_results[[i]]), 
                             Slope          = NA,
                             Slope_2.5      = NA,
                             Slope_97.5     = NA,
                             Std_err        = NA,
                             NumPositive    = NA,
                             NumNegative    = NA)
    
    for (j in seq_along(glmm_results[[i]])){
      M0_summary_sub <- glmm_results[[i]][[j]]
      if(class(M0_summary_sub) ==  "data.frame"){
        
        #M0_summary[j,-1] <- apply(M0_summary_sub[,-1],2,median, na.rm= T)
        M0_summary[j, "Slope"]      <- median(M0_summary_sub[,"Slope"], na.rm = T)
        #M0_summary[j, "Slope_2.5"]  <- quantile(M0_summary_sub[,"Slope"], c(.025), na.rm = T)
        #M0_summary[j, "Slope_97.5"] <- quantile(M0_summary_sub[,"Slope"], c(.975), na.rm = T)
        M0_summary[j, "Slope_2.5"]      <- median(M0_summary_sub[,"Slope_2.5"], na.rm = T)
        M0_summary[j, "Slope_97.5"]      <- median(M0_summary_sub[,"Slope_97.5"], na.rm = T)
        M0_summary[j, "Std_err"]    <- median(M0_summary_sub[,"Std.Error"], na.rm = T)
        
        if(!all(is.na(M0_summary_sub[,"Slope"]))){
          M0_summary[j,"NumPositive"] <- length(which(M0_summary_sub[,"Slope"] >0))
          M0_summary[j,"NumNegative"] <- length(which(M0_summary_sub[,"Slope"] <0))
          hist(M0_summary_sub[,"Slope"], main = paste(names(glmm_results)[i],names(glmm_results[[i]])[j]))
        }
      }
    }
    trait_taxa[[i]] <- M0_summary
    
  }
  names(trait_taxa)[which(names(trait_taxa) %in% "RELATIVE_OFFSPRING_SIZE")] <- "OFFSPRING_SIZE" 
  names(trait_taxa)[which(names(trait_taxa) %in% "RELATIVE_ALTITUDINAL_RANGE")] <- "ALTITUDINAL_RANGE"
  return(trait_taxa)
}

#' Extracts the slope and the conf. intervals of a glmmTMB model
#' @returns a vector with the three estimates
extract_effect_sizes <- function(glmmTMBModel){
  if (class(glmmTMBModel) == "glmmTMB") {
    slope <- confint(glmmTMBModel)[2,3]
    lower.conf.val <- confint(glmmTMBModel)[2,1]
    upper.conf.val <- confint(glmmTMBModel)[2,2]
    std.err        <- coefficients(summary(glmmTMBModel))$cond[2,2]
    
    return(c(slope, lower.conf.val, upper.conf.val, std.err))
  }
  return(NULL)
}

#' Residuals of altitude range with geographical range size (Linear regression, raw residuals)
get_alt_ran_resid <- function(dataset){
  if(nrow(na.omit(dataset[,c("ALTITUDINAL_RANGE", "GEOGRAPHICAL_RANGE_SIZE")])) == 0){
    dataset$RELATIVE_ALTITUDINAL_RANGE <- rep(NA, nrow(dataset))
  }else {
    ALTITUDINAL_RANGE <- log(dataset$ALTITUDINAL_RANGE)
    GEOGRAPHICAL_RANGE_SIZE <- log(dataset$GEOGRAPHICAL_RANGE_SIZE)
    RESIDS <- resid(lm(ALTITUDINAL_RANGE ~ GEOGRAPHICAL_RANGE_SIZE, na.action = "na.omit"))
    d <- rep(NA, length(dataset[,1]))
    for (i in 1:length(RESIDS)){
      d[as.numeric(names(RESIDS)[i])] <- RESIDS[i, drop = T]
    }
    dataset$RELATIVE_ALTITUDINAL_RANGE <- d
  }
  return(dataset)
}


#' Residuals of offspring size with body size (Linear regression, raw residuals)
get_offsize_resid <- function(dataset){
  if(nrow(na.omit(dataset[,c("OFFSPRING_SIZE", "BODY_SIZE")])) == 0){
    dataset$RELATIVE_OFFSPRING_SIZE <- rep(NA, nrow(dataset))
  }else {
    EGG_SIZE <- log(dataset$OFFSPRING_SIZE)
    BODY_SIZE <- log(dataset$BODY_SIZE)
    RESIDS <- resid(lm(EGG_SIZE ~ BODY_SIZE, na.action = "na.omit"))
    d <- rep(NA, length(dataset[,1]))
    for (i in 1:length(RESIDS)){
      d[as.numeric(names(RESIDS)[i])] <- RESIDS[i, drop = T]
    }
    dataset$RELATIVE_OFFSPRING_SIZE <- d
  }
  return(dataset)
}

#' Extracts a small set of information from meta_analyses objects into a dataframe
tablify_my_meta_analyses <- function(meta_objects){
  library(metafor)
  #create empty dataframe to store summary output
  fe_models    <- data.frame(Trait = names(meta_objects),
                             Estimate = NA, 
                             cil = NA,
                             ciu = NA, 
                             pval = NA,
                             `Groups` = NA)
  fe_models$Trait <- factor(fe_models$Trait, ordered = TRUE, 
                            levels = rev(fe_models$Trait))
  
  # Collect summary info
  for (i in seq_along(meta_objects)){
    if (!class(meta_objects[[i]][1]) == "logical"){ # meaning, not NA
      fe_models[i,"Estimate"] <- coef(meta_objects[[i]])
      fe_models[i,"pval"]     <- meta_objects[[i]]$pval
      fe_models[i,"cil"]      <- meta_objects[[i]]$ci.lb
      fe_models[i,"ciu"]      <- meta_objects[[i]]$ci.ub
      fe_models[i, "Groups"]       <- meta_objects[[i]]$k
    }
  }
  return(fe_models)
}
    
#### 2. DATA PREPARATION ####

#### 2.1 Import/Filter/Structure data ####
library(readxl)
library(ggplot2)
Dataset <- read_excel("20220503_Dataset.xlsx", sheet = "Dataset")
log_rules <- read_excel("20220503_Dataset.xlsx", sheet = "log_rules")

## Random terms ##
random_terms <- c( # VERTEBRATES        P /C /O /F /G
  "+ (1| Family)" # mammals,      1 /1 /12/32/53
  ,"+ (1| Family)" # birds,       1 /1 /12/39/58
  ,"+ (1| Family)" # reptiles,          1 /1 /2 /27/55
  ,"+ (1| Family)" # amphibians,        1 /1 /2 /21/48
  ,"+ (1| Family)" # fishes,      1 /1 /11/27/52
  # INVERTEBRATES                       P /C /O /F /G
  ,"+ (1| Family)" # Dragonflies,       1 /1 /1 /17/50
  ,"+ (1| Family)" # Butterflies, 1 /1 /1 /5 /33
  ,"+ (1| Family)" # Grasshoppers,      1 /1 /1 /10/46
  ,"+ (1| Family)" # Spiders,           1 /1 /1 /16/25
  ,"+ (1| Family)" # Gastropods,        1 /1 /3 /21/43
  # PLANTS                              P /C /O /F /G
  ,"+ (1| Family)" # Bryophytes,  2 /3 /14/33/47
  ,"+ (1| Family)" # Ferns,       1 /2 /6 /15/32
  ,"+ (1| Family)"       # Gymnosperms, 1 /3 /4 /9 /25
  ,"+ (1| Family)" # Monocots,    1 /1 /7 /14/45
  ,"+ (1| Family)"        # Legumes,     1 /1 /1 /1 /41
)

# Names of final traits to use
traits <- c("BODY_SIZE","RELATIVE_OFFSPRING_SIZE","FECUNDITY","GENERATION_LENGTH","DIET_BREADTH",
            "TROPHIC_LEVEL","DISPERSAL_ABILITY","MICROHABITAT",
            "HABITAT_BREADTH", "RELATIVE_ALTITUDINAL_RANGE")

# Subset Database into a list of 15 dataframes, one per major taxon

raw_traits <- c("BODY_SIZE","OFFSPRING_SIZE","FECUNDITY",
                "GENERATION_LENGTH","DIET_BREADTH",
                "TROPHIC_LEVEL","DISPERSAL_ABILITY",
                "MICROHABITAT","HABITAT_BREADTH","ALTITUDE_MIN",
                "ALTITUDE_MAX","GEOGRAPHICAL_RANGE_SIZE")
taxa <- c("Mammals", "Birds", "Reptiles", "Amphibians", "Fishes",
          "Dragonflies", "Butterflies", "Grasshoppers", "Spiders",
          "Snails", "Bryophytes","Ferns",  "Gymnosperms",
          "Monocots", "Legumes")
Group              <- Dataset[-1, "Group"]
Species            <- Dataset[-1, "IUCNName"]
Kingdom            <- Dataset[-1, "Kingdom"]
Phylum             <- Dataset[-1, "Phylum"]
Class              <- Dataset[-1, "Class"]
Order              <- Dataset[-1, "Order"]
Family             <- Dataset[-1, "Family"]
Genus              <- Dataset[-1, "Genus"]
Status             <- Dataset[-1, "Status"]
trait_columns      <- Dataset[-1,colnames(Dataset) %in% raw_traits]
# turn character into numeric
for (i in 1:ncol(trait_columns)){
  trait_columns[,i] <- as.numeric(as.character(trait_columns[[i]]))
}
#join together
info <- cbind(Species, Kingdom, Phylum, Class, Order, Family, 
              Genus, Status, trait_columns)
names(info) <- c("IUCNName", "Kingdom", "Phylum","Class", "Order",
                 "Family", "Genus", "Status", raw_traits)
# each taxa a separate data frame in a list
all <- list()
for (i in 1:length(taxa)){
  all[[i]] <- info[which(Group == taxa[i]),]
}
names(all) <- taxa



log_rules <- quick_clean(as.data.frame(log_rules))
#beanplots <- quick_clean(as.data.frame(beanplots))

### Trait resolutions #

traitRes_columns      <- Dataset[-1,which(colnames(Dataset) %in% traits) + 3]
traitRes_columns
names(traitRes_columns) <- colnames(Dataset)[which(colnames(Dataset) %in% traits)]

res <- cbind(Group, Species, traitRes_columns)
res["NA",] <- NA

# each taxa a separate data frame in a list
all_res <- list()
for (i in 1:length(taxa)){
  all_res[[i]] <- res[which(Group == taxa[i]),]
}
names(all_res) <- taxa


#### 2.2. Derived variables ####

#' Binary response variable (threatened vs unthreatened)
all2 <- lapply(all, function(x){
  b <- x$Status
  x1 <-x
  b[x$Status %in% c("NT", "VU", "EN", "CR", "EW", "EX", "CR(PE)")] <- "TH"
  x1$StatusB <- b
  return(x1)
})

#' Binary response (threatened (1) vs unthreatened (0))
all2 <- lapply(all2, function(x){
  x1 <- x
  x1$StatusC <- ifelse(x$StatusB == "TH", 1, 0)
  return(x1)
})

# Altitude range
all2 <- lapply(all2, function(x){
  x1 <- x
  x1$ALTITUDINAL_RANGE <- x$ALTITUDE_MAX - x$ALTITUDE_MIN
  x1$ALTITUDINAL_RANGE <- x1$ALTITUDINAL_RANGE + 10 #see methods
  return(x1)
})

all2 <- lapply(all2, get_alt_ran_resid)

all2 <- lapply(all2, get_offsize_resid)

# Resolution of the trait RELATIVE_OFFSPRING_SIZE
for (i in 1:length(all_res)){
  body_size      <- all_res[[i]]$BODY_SIZE
  offspring_size <- all_res[[i]]$OFFSPRING_SIZE
  RELATIVE_OFFSPRING_SIZE <- body_size

  RELATIVE_OFFSPRING_SIZE <- body_size
  
  key <- cbind(c("species", "close species", "genus", "family", "NA"))
  
  for (j in seq_along(body_size)){
    bs <- which(key == body_size[j])
    os <- which(key == offspring_size[j])
    RELATIVE_OFFSPRING_SIZE[j] <- max(bs,os)
    RELATIVE_OFFSPRING_SIZE[j] <- key[as.numeric(RELATIVE_OFFSPRING_SIZE[j])]
  }
  all_res[[i]]$RELATIVE_OFFSPRING_SIZE <- RELATIVE_OFFSPRING_SIZE
}


# #shameless copy paste of code:
# # Resolution of the trait RELATIVE_ALTITUDINAL_RANGE
# for (i in 1:length(all_res)){
#   geographical_range_size  <- all_res[[i]]$GEOGRAPHICAL_RANGE_SIZE
#   altitudinal_range        <- all_res[[i]]$ALTITUDINAL_RANGE
#   RELATIVE_ALTITUDINAL_RANGE <- geographical_range_size
#   
#   RELATIVE_ALTITUDINAL_RANGE <- geographical_range_size
#   
#   key <- cbind(c("species", "close species", "genus", "family", "NA"))
#   
#   for (j in seq_along(geographical_range_size)){
#     bs <- which(key == geographical_range_size[j])
#     os <- which(key == altitudinal_range[j])
#     RELATIVE_ALTITUDINAL_RANGE[j] <- max(bs,os)
#     RELATIVE_ALTITUDINAL_RANGE[j] <- key[as.numeric(RELATIVE_ALTITUDINAL_RANGE[j])]
#   }
#   all_res[[i]]$RELATIVE_ALTITUDINAL_RANGE <- RELATIVE_ALTITUDINAL_RANGE
# }


#### 2.3 Log-transform traits ####

all_log <- log_transform(all2, log_rules)


#*************************************#

#### 3. ANALYSES ###############

#### 3.1 Generalized linear mixed models #### 

glmm_results      <- run_my_glmms(all_log, traits, random_terms, 3)

#### 3.2 Meta-analyses #### 

glmm_results_summary <- glmmTMB_to_Dataframe(glmm_results)
vertebrates <- c(1:5)
invertebrates <- c(6:10)
plants <- c(11:15)
glmm_results_summary_v <-lapply(glmm_results_summary, function(x) x[vertebrates,])
glmm_results_summary_i <-lapply(glmm_results_summary, function(x) x[invertebrates,])
glmm_results_summary_p <-lapply(glmm_results_summary, function(x) x[plants,])

meta_objects   <- run_my_meta_analyses(glmm_results_summary)
meta_objects_v <- run_my_meta_analyses(glmm_results_summary_v)
meta_objects_i <- run_my_meta_analyses(glmm_results_summary_i)
meta_objects_p <- run_my_meta_analyses(glmm_results_summary_p)

#### 4. PRESENTING RESULTS ####

#### 4.1 Table S2: Number of observations per trait-taxa ####

# Observations
p <- lapply(glmm_results_summary, function(x) {
  z <- ifelse(x$Convergence == 0, x$Observations, 0)
  names(z) <- x$Taxa
  return(z)
})
p
c <- c()
for (i in seq_along(glmm_results)){
  c <- cbind(c, p[[i]])
}
colnames(c) <- names(p)
c
#write.table(c, "clipboard",  sep = "\t")

#### 4.2 % imputed values per trait:taxa #####

# Imputed values
Imputed <- vector("list", length(glmm_results[1:8])) #only up to microhabitat
names(Imputed) <- names(glmm_results)[1:8]

for (i in seq_along(glmm_results[1:8])){
  M0_summary <- data.frame(Taxa = names(all_log), 
                           ImputedObs    = NA)
  for (j in seq_along(glmm_results[[i]])){
    if (class(glmm_results[[i]][[j]]) == "glmmTMB"){
      ImputedObs <- na.exclude(
        all_res[[j]][which(all_res[[j]][,traits[i]] != "species"),traits[i]])
      ImputedObs <- length(ImputedObs[ImputedObs != "NA"])
      ImputedObs <- ifelse(is.na(ImputedObs), 0, ImputedObs)
      M0_summary[j, "ImputedObs"]   <- ImputedObs
    }
  }
  Imputed[[i]] <- M0_summary
}

p <- lapply(Imputed, function(x) {
  z <- x$ImputedObs
  names(z) <- x$Taxa
  return(z)
})
p
c1 <- c()
for (i in seq_along(Imputed)){
  c1 <- cbind(c1, p[[i]])
}
colnames(c1) <- names(p)
c1

# % Imputed vals
proportion_imputed <- round(c1 / c[1:8] * 100,1)
rowMeans(proportion_imputed, na.rm = T)
colMeans(proportion_imputed, na.rm = T)
mean(proportion_imputed, na.rm = T)

write.table(proportion_imputed, "clipboard",row.names = T ,col.names = T,sep = "\t")

#### 4.3 Fig. 1 ####

trait_labs <- c("Body size","Offspring size","Fecundity","Generation length","Diet breadth",
                "Trophic level","Dispersal ability","Microhabitat",
                "Habitat breadth", "Altitudinal range")

plot_nice_figure(glmm_results_summary, list(meta_objects_v, meta_objects_i , meta_objects_p),
                 trait_labs = trait_labs, file_name = "fig_1original.jpg", dpi = 500, device = "jpg")

#### 5. Additional analyses ####

#### 5.1. Fig. S1 Without imputed values ####

# Make dataset excluding imputed values
traits
all_log_excl <- all_log
for (i in 1:length(all_log_excl)){
  my_cols1 <- which(names(all_log_excl[[i]]) %in% traits)
  my_cols2 <- which(names(all_res[[i]]) %in% traits)
  for (j in 1:length(my_cols1)){
    for (k in 1:nrow(all_log_excl[[i]])){
      this_cell_res <- all_res[[i]][k,my_cols2[j]]
      if(!is.na(this_cell_res) && !is.null(this_cell_res) && this_cell_res != "species")
        all_log_excl[[i]][k,my_cols1[j]] = NA
    }
  }
}
#it is stored here:
all_log_excl

# Now run the models on this
imput <- run_my_glmms(all_log_excl, traits, random_terms, 3)
imput_summary <-  glmmTMB_to_Dataframe(imput)
imput_summary_v <-lapply(imput_summary, function(x) x[vertebrates,])
imput_summary_i <-lapply(imput_summary, function(x) x[invertebrates,])
imput_summary_p <-lapply(imput_summary, function(x) x[plants,])

imput_meta   <- run_my_meta_analyses(imput_summary)
imput_meta_v <- run_my_meta_analyses(imput_summary_v)
imput_meta_i <- run_my_meta_analyses(imput_summary_i)
imput_meta_p <- run_my_meta_analyses(imput_summary_p)

plot_nice_figure(imput_summary[1:8], list(imput_meta_v[1:8], imput_meta_i[1:8],imput_meta_p[1:8]),
                 trait_labs = trait_labs[1:8], file_name = "fig_S1without_imputation.jpg",dpi = 500, device = "jpg")

#### 5.2 Fig. S2 Test of differences in sample size ####
# Run GLMMs with bootstrapped datasets
if(run_bootstraps){
  spsize <- run_my_glmms_bootstrap_df(all_log, traits = traits,
                                      random_terms = random_terms,
                                      min.obs.level = 3, reps = 100)
  spsize_summary <- glmmTMB_to_Dataframe2(spsize)

  
}else
  spsize_summary <- readRDS(file.path("spsize_summary"))

spsize_summary_v <-lapply(spsize_summary, function(x) x[vertebrates,])
spsize_summary_i <-lapply(spsize_summary, function(x) x[invertebrates,])
spsize_summary_p <-lapply(spsize_summary, function(x) x[plants,])

spsize_meta   <- run_my_meta_analyses(spsize_summary)
spsize_meta_v <- run_my_meta_analyses(spsize_summary_v)
spsize_meta_i <- run_my_meta_analyses(spsize_summary_i)
spsize_meta_p <- run_my_meta_analyses(spsize_summary_p)

plot_nice_figure(spsize_summary, list(spsize_meta_v, spsize_meta_i,spsize_meta_p),
                 trait_labs = trait_labs, file_name = "fig_S2bootstrap.jpg",dpi = 500, device = "jpg")

#### 5.3 Fig. S3 Comprehensive datasets of mammals and birds ####

more_data <- all2
my_birds <- read.csv("my_birds.csv")
more_data$Birds <- my_birds
my_mammals <- read.csv("my_mammals.csv")
more_data$Mammals <- my_mammals

#' Residuals of altitude range with geographical range size (Linear regression, raw residuals)
more_data <- lapply(more_data, get_alt_ran_resid)

#' Residuals of offspring size with body size (Linear regression, raw residuals)
more_data <- lapply(more_data, get_offsize_resid)
# log transform
more_data <- log_transform(more_data, log_rules)

moredata <- run_my_glmms(more_data[1:2], traits = traits,
                                    random_terms = random_terms[1:2],
                                    min.obs.level = 3)
moredata_summary <- glmmTMB_to_Dataframe(moredata)

#new labels for traits
plot_nice_figure(moredata_summary, NULL,
                 trait_labs = trait_labs, file_name = "fig_S3moredata.jpg", width = 10,dpi = 500, device = "jpg")


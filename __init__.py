__version__=0.47

### Load the FairDetect Class
import time

start_time = time.time()

import matplotlib.pyplot as plt
from random import randrange
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import dalex as dx
from scipy.stats import chi2_contingency
from tabulate import tabulate
from sklearn.metrics import confusion_matrix
from scipy.stats import chisquare
from sklearn.metrics import precision_score


class Fairdetect:
    
    def __init__(self,model,X_test,y_test):
        self.model,self.X_test,self.y_test = model, X_test, y_test
    
    def check_for_target(self,data_bd):
        """
        Allows user to define the target variable out of the columns available
        Checks if target variable is Binary or not. Only binary target variable accepted
        
        Return
        ------
        Selection of the target variable from all columns of the dataframe
        """
        print(data_bd.columns.values)
        data_bd.target_column = input("Please select target column")
        if not(set(data_bd[data_bd.target_column].unique())==set([0,1])):
            raise ValueError('Only binary target variable')
        else:
            print("You choose wisely. Your target variable is:", data_bd.target_column)
        
        return data_bd.target_column
    
    
    def create_labels(self,sensitive):
        """
        Define labels for sensitive variable
        
        Parameters
        ----------
        create_labels: Method allows user to input a string for each group of sensitive variable
        """
        sensitive_label = {}
        for i in set(self.X_test[sensitive]):
            text = "Please Enter Label for Group" +" "+ str(i)+": "
            label = input(text)
            sensitive_label[i]=label
        return(sensitive_label)
    
    def create_targetname(self,predictions):
        """
        Define labels for sensitive variable
        
        Parameters
        ----------
        create_targetname: Method allows user to input a string for each group of target variable
        """
        print()
        print("Enter Target names below") 
        target_name = {}
        for i in set(predictions):
            text = "Please Enter name for target predicted" +" "+ str(i)+": "
            name = input(text)
            target_name[i]=name
            #print(target_name)
        return(target_name)
    
    def get_sensitive_col(self):
        """
        Allows user to define the sensitive variable out of the columns available
        Checks if sensitive variable is Binary or not. Only binary sensitive variable accepted
        
        Return
        ------
        Selection for the sensitive variable from all columns of the dataframe
        """     
        
        print("Please select the sensitive column name from given features")
        print(self.X_test.columns.values)
        sens_col=input("Enter sensitive column here : ")
        print()
        return sens_col
    
    def representation(self,X_test,y_test,sensitive,labels,predictions,target_names):
        """
        Identifies representation bias for a relationship between the sensitive group and the target group
        
        Parameters
        ----------
        Model: Test dataset
        Sensitive: Defined sensitive variable to be analyzed.
        Labels: Labels definition for the sensitive variable
        Predictions: Predicted values based on train dataset
        
        Returns
        -------   
        Contingency table with % distribution of the data set between both sensitive and target variables
        Bar charts supporting the contingency table
        P-value result of the hypothesis H0: No Significant representation disparity, indicating the existance 
        or not of a relationship between the sensitive and target variables
              
        Notes
        ------
        The p-values are calculated using the Chi test-statistic
        """

        full_table = X_test.copy()
        sens_df = {}

        for i in labels:
            full_table['p'] = predictions
            full_table['t'] = y_test
            sens_df[labels[i]] = full_table[full_table[sensitive]==i]
        
        contigency_p = pd.crosstab(full_table[sensitive], full_table['t']) 
        cp, pp, dofp, expectedp = chi2_contingency(contigency_p) 
        contigency_pct_p = pd.crosstab(full_table[sensitive], full_table['t'], normalize='index')

        
        sens_rep = {}
        for i in labels:
            sens_rep[labels[i]] = (X_test[sensitive].value_counts()/X_test[sensitive].value_counts().sum())[i]

        labl_rep = {}
        for i in labels:
            labl_rep[str(i)] = (y_test.value_counts()/y_test.value_counts().sum())[i]


        fig = make_subplots(rows=1, cols=2)

        for i in labels:
            fig.add_trace(go.Bar(
            showlegend=False,
            x = [labels[i]],
            y= [sens_rep[labels[i]]]),row=1,col=1)

            fig.add_trace(go.Bar(
            showlegend=False,
            x = [target_names[i]],
            y= [labl_rep[str(i)]],
            marker_color=['orange','blue'][i]),row=1,col=2)

        c, p, dof, expected = chi2_contingency(contigency_p)
        cont_table = (tabulate(contigency_pct_p.T, headers=labels.values(),showindex= target_names.values(), tablefmt='fancy_grid'))

        return cont_table, sens_df, fig, p
        

    
    def ability(self,sens_df,labels):
        """
        Method to calculate ability bias for the sensitive group members
        
        Parameters
        ----------
        sens_df: model results df for based on the sensitive variable definition
        Labels: Labels definition for the sensitive variable
        
        Returns
        -------   
        TPR: True Positive Rate calculation
        FPR: False Positive Rate calculation
        TNR: True Negative Rate calculation 
        FNR: False Negative Rate calculation        
        """ 
        sens_conf = {}
        for i in labels:
            sens_conf[labels[i]] = confusion_matrix(list(sens_df[labels[i]]['t']), list(sens_df[labels[i]]['p']), labels=[0,1]).ravel()

        true_positive_rate = {}
        false_positive_rate = {}
        true_negative_rate = {}
        false_negative_rate = {}

        for i in labels:
            true_positive_rate[labels[i]] = (sens_conf[labels[i]][3]/(sens_conf[labels[i]][3]+sens_conf[labels[i]][2]))
            false_positive_rate[labels[i]] = (sens_conf[labels[i]][1]/(sens_conf[labels[i]][1]+sens_conf[labels[i]][0]))
            true_negative_rate[labels[i]] = 1 - false_positive_rate[labels[i]]
            false_negative_rate[labels[i]] = 1 - true_positive_rate[labels[i]]

        return(true_positive_rate,false_positive_rate,true_negative_rate,false_negative_rate)



    def ability_plots(self,labels,TPR,FPR,TNR,FNR):
        """
        Method to plot the results of the ability method leveraging bar charts
        
        Parameters
        ----------
        Labels: Labels definition for the sensitive variable
        TPR: True Positive Rate calculation result from abilty method 
        FPR: False Positive Rate calculation result from abilty method
        TNR: True Negative Rate calculation result from abilty method
        FNR: False Negative Rate calculation result from abilty method
        
        Returns
        -------   
        Bar charts supporting each of the scenarios above split by defined sensitive variable          
        """
        
        fig = make_subplots(rows=2, cols=2, 
                            subplot_titles=("True Positive Rate", "False Positive Rate", "True Negative Rate", "False Negative Rate"))

        x_axis = list(labels.values())
        fig.add_trace(
            go.Bar(x = x_axis, y=list(TPR.values())),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x = x_axis, y=list(FPR.values())),
            row=1, col=2
        )

        fig.add_trace(
            go.Bar(x = x_axis, y=list(TNR.values())),
            row=2, col=1
        )

        fig.add_trace(
            go.Bar(x = x_axis, y=list(FNR.values())),
            row=2, col=2
        )

        fig.update_layout(showlegend=False,height=600, width=800, title_text="Ability Disparities")
        fig.show()


    def ability_metrics(self,TPR,FPR,TNR,FNR):
        """
        Method to calculate the Chi square test and p-value for each of the scenarios defined in the
        ability method
        
        Parameters
        ----------
        TPR: True Positive Rate calculation result from ability method 
        FPR: False Positive Rate calculation result from ability method
        TNR: True Negative Rate calculation result from ability method
        FNR: False Negative Rate calculation result from ability method
        
        Returns
        -------   
        P-value result of the hypothesis H0: No Significant ability bisparity. P-value is calculated 
        for each of the scenarios indicating level of disparity based on the following criteria in the case 
        of identifying disparity:
        
        If p-value <= 0.01 -> ***
        If p-value <= 0.05 -> **
        If p-value <= 0.1 -> *
        
        Cases in which the null hypothesis H0 is rejected are printed bold and underlined. 
        
        Notes
        ------
        The p-values are calculated using the Chi test-statistic         
        """
            
        self.TPR_p = chisquare(list(np.array(list(TPR.values()))*100))[1]
        self.FPR_p = chisquare(list(np.array(list(FPR.values()))*100))[1]
        self.TNR_p = chisquare(list(np.array(list(TNR.values()))*100))[1]
        self.FNR_p = chisquare(list(np.array(list(FNR.values()))*100))[1]

        if self.TPR_p <= 0.01:
            print('\033[1;4m' +"*** Reject H0: Significant True Positive Disparity with p="+'\033[0m',self.TPR_p)
        elif self.TPR_p <= 0.05:
            print('\033[1;4m' +"** Reject H0: Significant True Positive Disparity with p="+'\033[0m',self.TPR_p)
        elif self.TPR_p <= 0.1:
            print('\033[1;4m' +"*  Reject H0: Significant True Positive Disparity with p="+'\033[0m',self.TPR_p)
        else:
            print("Accept H0: True Positive Disparity Not Detected. p=",self.TPR_p)

        if self.FPR_p <= 0.01:
            print('\033[1;4m' +"*** Reject H0: Significant False Positive Disparity with p="+'\033[0m',self.FPR_p)
        elif self.FPR_p <= 0.05:
            print('\033[1;4m' +"** Reject H0: Significant False Positive Disparity with p="+'\033[0m',self.FPR_p)
        elif self.FPR_p <= 0.1:
            print('\033[1;4m' +"*  Reject H0: Significant False Positive Disparity with p="+'\033[0m',self.FPR_p)
        else:
            print("Accept H0: False Positive Disparity Not Detected. p=",self.FPR_p)

        if self.TNR_p <= 0.01:
            print('\033[1;4m' +"*** Reject H0: Significant True Negative Disparity with p="+'\033[0m',self.TNR_p)
        elif self.TNR_p <= 0.05:
            print('\033[1;4m' +"** Reject H0: Significant True Negative Disparity with p="+'\033[0m', self.TNR_p)
        elif self.TNR_p <= 0.1:
            print('\033[1;4m' +"*  Reject H0: Significant True Negative Disparity with p="+'\033[0m', self.TNR_p)
        else:
            print("Accept H0: True Negative Disparity Not Detected. p=",self.TNR_p)

        if self.FNR_p <= 0.01:
            print('\033[1;4m' +"*** Reject H0: Significant False Negative Disparity with p="+'\033[0m',self.FNR_p)
        elif self.FNR_p <= 0.05:
            print('\033[1;4m' +"** Reject H0: Significant False Negative Disparity with p="+'\033[0m',self.FNR_p)
        elif self.FNR_p <= 0.1:
            print('\033[1;4m' +"*  Reject H0: Significant False Negative Disparity with p="+'\033[0m',self.FNR_p)
        else:
            print("Accept H0: False Negative Disparity Not Detected. p=",self.FNR_p)


    def predictive(self,labels,sens_df):
        
        """
        This method calculates disparity between the orignal model and the predicted model
        
        Parameters
        ----------
        sens_df: model results df for based on the sensitive variable definition
        Labels: Labels definition for the sensitive variable
        
        Returns
        -------   
        Bar charts indicating predictive disparity level split by defined sensitive variable 
        P-value result of the hypothesis H0: No Significant Predictive Disparity
        
        Notes
        ------
        The p-values are calculated using the Chi test-statistic         
        """
        precision_dic = {}

        for i in labels:
            precision_dic[labels[i]] = precision_score(sens_df[labels[i]]['t'],sens_df[labels[i]]['p'])

        fig = go.Figure([go.Bar(x=list(labels.values()), y=list(precision_dic.values()))])

        pred_p = chisquare(list(np.array(list(precision_dic.values()))*100))[1]

        return(precision_dic,fig,pred_p)

    
    def identify_bias(self, sensitive,labels):
        """
        This method summarizes the areas of bias within FairDetect framework.
        
        Parameters
        ----------
        Sensitive: Defined sensitive variable to be analyzed.
        Labels: Labels definition for the sensitive variable
        
        Returns
        -------
        Breakdown of Bias identification into 3 categories
        
        Representation: Representation of the sensitive variables and its association with 
        the target variable and the p-values associated in order to identify representation disparity
        
        Ability: Compares the ability for the sensitive group members
        
        Predictive: Compares the distribution within the data set and the prediction set
        
        Notes
        ------
        The p-values are calculated using the Chi test-statistic
        """
        predictions = self.model.predict(self.X_test)
        target_names=self.create_targetname(predictions)
        cont_table,sens_df,rep_fig,rep_p = self.representation(self.X_test,self.y_test,sensitive,labels,predictions,target_names)

        print("REPRESENTATION")
        rep_fig.show()

        print(cont_table,'\n')

        if rep_p <= 0.01:
            print('\033[1;4m' +"*** Reject H0: Significant Relation Between",sensitive,"and Target with p="+'\033[0m',rep_p)
        elif rep_p <= 0.05:
            print('\033[1;4m' +"** Reject H0: Significant Relation Between",sensitive,"and Target with p="+'\033[0m',rep_p)
        elif rep_p <= 0.1:
            print('\033[1;4m' +"* Reject H0: Significant Relation Between",sensitive,"and Target with p="+'\033[0m',rep_p)
        else:
            print("Accept H0: No Significant Relation Between",sensitive,"and Target Detected. p=",rep_p)

        TPR, FPR, TNR, FNR = self.ability(sens_df,labels)
        print("\n\nABILITY")
        self.ability_plots(labels,TPR,FPR,TNR,FNR)
        self.ability_metrics(TPR,FPR,TNR,FNR)


        precision_dic, pred_fig, pred_p = self.predictive(labels,sens_df)
        print("\n\nPREDICTIVE")
        pred_fig.show()

        if pred_p <= 0.01:
            print('\033[1;4m'+"*** Reject H0: Significant Predictive Disparity with p="+'\033[0m',pred_p)
        elif pred_p <= 0.05:
            print('\033[1;4m'+"** Reject H0: Significant Predictive Disparity with p="+'\033[0m',pred_p)
        elif pred_p <= 0.1:
            print('\033[1;4m'+"* Reject H0: Significant Predictive Disparity with p="+'\033[0m',pred_p)
        else:
            print("Accept H0: No Significant Predictive Disparity. p=",pred_p)
    
 
    def disparate_impact(self,sensitive, labels):
        
        """
        Disparate impact is the ratio of unfavoured/favoured groups only considering occurences with the desired outcome
        
        Parameters
        ----------
        Sensitive: Defined sensitive variable to be analyzed.
        Labels: Labels definition for the sensitive variable
        
        Returns
        -------
        Plots for each one of the available combinations sensitive vs. target variables
        
        """
        
        actual_test_table = self.X_test.copy()
        actual_test_table['t'] = self.y_test
        actual_test_table['p'] = self.model.predict(self.X_test)
        
        favoured_df = actual_test_table[actual_test_table[sensitive] == 0]
        num_of_favoured = favoured_df.shape[0]
        unfavoured_df = actual_test_table[actual_test_table[sensitive] == 1]
        num_of_unfavoured = unfavoured_df.shape[0]
        
        positive_outcomes_favoured = favoured_df[favoured_df['t'] ==1].shape[0]
        positive_ratio_favoured = positive_outcomes_favoured/num_of_favoured
        
        positive_outcomes_unfavoured = unfavoured_df[unfavoured_df['t'] ==1].shape[0]
        positive_ratio_unfavoured = positive_outcomes_unfavoured/num_of_unfavoured
                           
        self.disparate_impact = positive_ratio_unfavoured / positive_ratio_favoured
        print("Disparate Impact, Sensitive vs. Predicted Target: " + str(self.disparate_impact))
       
        
        if self.disparate_impact <= 0.8:
            print('\033[1;4m'+"The disparate impact ratio is below 0.8 indicating a favour towards the priviliged group"+'\033[0m')
        elif self.disparate_impact <= 0.9:
            print("The disparate impact ratio indicates a slight favour towards the priviliged group")
        else: 
            print("The disparate impact ratio indicated complete equality between the favoured and unfavoured group")
            
        plt.bar('Ratio',self.disparate_impact)
        plt.title('Disparate impact')
        plt.xlabel('Ratio for postive outcomes for favoured and unfavoured groups')
        plt.ylabel('disparate_impact')
        plt.show()
        return(self.disparate_impact)

    def understand_shap(self,labels,sensitive,affected_group,affected_target):
        
        """
        This method supports analysis of the differentiate the bias on the affected group vs. the overall population 
        and looks to understand what variables are the source of biggest disparity
        
        Parameters
        ----------
        Sensitive: Defined sensitive variable to be analyzed.
        Labels: Labels definition for the sensitive variable
        Affected_group: defined impacted group as per sensitive variable definition
        Affected_target: defined unpriviledge group as per target variable definition
        
        Returns
        -------
        Plots for each one of the available combinations sensitive vs. target variables compared to the base scenario: 
        variable relevance for all sensitive groups and all targets
        """       
       
        import shap
        explainer = shap.Explainer(self.model)

        full_table = self.X_test.copy()
        full_table['t'] = self.y_test
        full_table['p'] = self.model.predict(self.X_test)
        full_table

        shap_values = explainer(self.X_test)
        sens_glob_coh = np.where(self.X_test[sensitive]==list(labels.keys())[0],labels[0],labels[1])

        misclass = full_table[full_table.t != full_table.p]
        affected_class = misclass[(misclass[sensitive] == affected_group) & (misclass.p == affected_target)]
        shap_values2 = explainer(affected_class.drop(['t','p'],axis=1))

        figure,axes = plt.subplots(nrows=2, ncols=2,figsize=(20,10))
        plt.subplots_adjust(right=1.4,wspace=1)

        print("Model Importance Comparison")
        plt.subplot(1, 2, 1) # row 1, col 2 index 1
        shap.plots.bar(shap_values.cohorts(sens_glob_coh).abs.mean(0),show=False)
        plt.subplot(1, 2, 2) # row 1, col 2 index 1
        shap_values2 = explainer(affected_class.drop(['t','p'],axis=1))
        shap.plots.bar(shap_values2)

        full_table['t'] = self.y_test
        full_table['p'] = self.model.predict(self.X_test)

        misclass = full_table[full_table.t != full_table.p]
        affected_class = misclass[(misclass[sensitive] == affected_group) & (misclass.p == affected_target)]

        truclass = full_table[full_table.t == full_table.p]
        tru_class = truclass[(truclass[sensitive] == affected_group) & (truclass.t == affected_target)]

        x_axis = list(affected_class.drop(['t','p',sensitive],axis=1).columns)
        affect_character = list((affected_class.drop(['t','p',sensitive],axis=1).mean()-tru_class.drop(['t','p',sensitive],axis=1).mean())/affected_class.drop(['t','p',sensitive],axis=1).mean())
        fig = go.Figure([go.Bar(x=x_axis, y=affect_character)])

        print("Affected Attribute Comparison")
        print("Average Comparison to True Class Members")
        fig.show()

        misclass = full_table[full_table.t != full_table.p]
        affected_class = misclass[(misclass[sensitive] == affected_group) & (misclass.p == affected_target)]

        tru_class = full_table[(full_table[sensitive] == affected_group) & (full_table.p == affected_target)]

        x_axis = list(affected_class.drop(['t','p',sensitive],axis=1).columns)
        affect_character = list((affected_class.drop(['t','p',sensitive],axis=1).mean()-full_table.drop(['t','p',sensitive],axis=1).mean())/affected_class.drop(['t','p',sensitive],axis=1).mean())

        fig = go.Figure([go.Bar(x=x_axis, y=affect_character)])
        print("Average Comparison to All Members")
        fig.show()

        print("Random Affected Decision Process")
        explainer = shap.Explainer(self.model)
        shap.plots.waterfall(explainer(affected_class.drop(['t','p'],axis=1))[randrange(0, len(affected_class))],show=False)
        
end_time = time.time()

print("time taken in execution is :", end_time - start_time)
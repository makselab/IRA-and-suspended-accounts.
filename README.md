# IRA-and-suspended-accounts
The Role of the IRA in Twitter during the 2016 US Presidential Elections: Unveiling Amplification and Influence of Suspended Accounts

Analysis codes to reproduce the results of the paper:
M. Serafino, Z. Zhou, J.S. Andrade, A. Bovet, H. A. Makse 
"The Role of the IRA in Twitter during the 2016 US Presidential Elections: Unveiling Amplification and Influence of Suspended Accounts"

By using this code, you agree with the following points:

The code is provided without any warranty or conditions of any kind. We assume no responsibility for errors or omissions in the results and interpretations following from applications of the code.

You commit to cite our paper (above) in publications where you use this code.

The dataset containing the retweet networks and the tweet ids of the tweets used in the article is available here: see https://hmakse.ccny.cuny.edu/twitter-analysis/.
The IRA dataset can be found at: https://osf.io/g4hws/.

The repository is organized as follows: 
1) The utility files generate the main results.
   
   1.0) IRA_identify.py matches the IRA users from the IRA dataset released by Twitter in the 2016 dataset. general_utilities and GraphUtils contain functions called in the utilities below.
   
   1.1) Section_one_a_utilities.py generates the retweets networks per category ( full or sampled one, see paper).

   1.2) Section_one_b_utilities.py creates the files containing the information about the tweets in each category, such as source, datetime, tweet_id, etc.

   1.3) Section_two_utilities.py  created the interactions networks and aggregated networks ( both ego and expanded, see paper).

   1.4) Section_three_utilities.py prepare the data for the causal analysis. It collects the  supporting class activities and performs the STL filtering.

   1.5) Section_three__b_utilities.py perform the Granger causal analysis on the residuals computed by means of Section_three_utilities.py.

   
2) The folder notebooks contain the jupyter notebooks that guide you through the results and visualizations section by section.

3) The data folder contains the mapping of IRA users with the 2016 dataset (which you can obtain using  IRA_identify.py) as well as the news outlets used in the paper.
   
# Collective Influence algorithm
To compute the Collective Influence ranking of nodes in the retweet networks, you must first compile the cython code with : python setup.py build_ext --inplace

Check https://github.com/alexbovet/information_diffusion

# IRA-and-suspended-accounts
The Role of the IRA in Twitter during the 2016 US Presidential Elections: Unveiling Amplification and Influence of Suspended Accounts

Analysis codes to reproduce the results of the paper:
M. Serafino, Z. Zhou, J.S. Andrade, A. Bovet, H. A. Makse 
"The Role of the IRA in Twitter during the 2016 US Presidential Elections: Unveiling Amplification and Influence of Suspended Accounts"

By using this code, you agree with the following points:

The code is provided without any warranty or conditions of any kind. We assume no responsibility for errors or omissions in the results and interpretations following from applications of the code.

You commit to cite our paper (above) in publications where you use this code.

The dataset containing the retweet networks and the tweet ids of the tweets used in the article is available here:
TO ADD

The repository is organized as follows: 
1) The utility files generate the main results.
   1.1) Section_one_a_utilities.py  generates the retweets networks per category ( full or sampled one, see paper)
   1.2) 
3) The folder notebooks contain the jupyter notebooks that guide you through the results and visualizations.
   
# Collective Influence algorithm
To compute the Collective Influence ranking of nodes in the retweet networks, you must first compile the cython code with : python setup.py build_ext --inplace

Check https://github.com/alexbovet/information_diffusion

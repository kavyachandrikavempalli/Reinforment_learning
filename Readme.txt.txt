The following are the steps needed to run the reinforce and actor-critic algorithms.

1. Our project resides in Team_project.zip, Extract the file.
2. The code is based on the Individual Project-3 and the edits for team project are done in qlearningAgents.py file.
3. The ReinforceAgent class contains the code for REINFORCE algorithm
4. The ActorCriticAgent class contains the code for Actor-Critic Algorithm.
5. To run the REINFORCE algorithm, 
Run the following command in the command prompt, after moving to the code directory:

python pacman.py -p ReinforceAgent -x 500 -n 510 -l layout-name --frameTime 0

6. To run the Actor-Critic algorithm,
Run the following command in the command prompt, after moving to the code directory:

python pacman.py -p ActorCritic -x 500 -n 510 -l layout-name --frameTime 0

7. To generate custom layouts, run the following code:

python random_layout_producer.py

8. Run the ttest.py to generate t-test scores for the file
COMMON TASKS

1. HAMMER GAME (/hammergame)

This is a fairground-style interactive game where players try to stop a swinging needle at exactly 90 degrees to get the highest score. The swing moves between 0 and 180 degrees continuously. The closer the player stops it to 90 degrees, the higher the score. A perfect stop at 90 gives a score of 100.

The score decreases proportionally as the deviation from 90 increases. Realistic motion is simulated with variable speed: the swing is faster near 90 and slower near the edges. A two-player mode is also included, where each player takes turns and the game declares a winner based on the final score.

Technologies used: HTML, CSS, JavaScript

2. Polynomic Vault – Shamir’s Secret Sharing (Located in: sss.py)

This task implements Shamir’s Secret Sharing Scheme using polynomial-based cryptography. A secret number is split into multiple shares using a randomly generated polynomial. Only a minimum threshold number of shares are required to reconstruct the original secret.

The program includes both the share generation and reconstruction using Lagrange interpolation. The implementation is written in Python and demonstrates secure secret splitting and recovery with sample runs.

Technologies used: Python

3. Vehicle Image Classification (Located in: ml.py)

This machine learning task classifies vehicle images into one of seven categories: Auto Rickshaws, Bikes, Cars, Motorcycles, Planes, Ships, and Trains. The ResNet18 model from torchvision is used as the base model and fine-tuned to fit the custom dataset.

The dataset is loaded using PyTorch’s ImageFolder, and appropriate transformations such as resizing, normalization, and tensor conversion are applied. After training the model for a few epochs, the final test loss and accuracy are printed. Sample predictions on test images are also displayed using matplotlib.

Technologies used: Python, PyTorch, torchvision, matplotlib

DOMAIN SPECIFIC TASK 

APPLICATION DEVELOPMENT 

Disc Battle 4 (Located in: boardgame/)

This is a two-player strategy game inspired by Connect-4 with additional twists. Players alternate turns dropping red and yellow discs into a 6x7 grid. Before each move, the opponent blocks one column, and the current player cannot place a disc in that column.

The goal is to connect four of one’s own discs horizontally, vertically, or diagonally. A timer is implemented for each player, and if it runs out, the other player wins. A leaderboard system stores player statistics in local storage.

Additional features include power-ups (such as extra time or extra turn), sound effects with an on/off toggle, theme switching between dark and light modes, an undo function, and a complete game replay feature.

Technologies used: HTML, CSS, JavaScript


Author

Sakthi M
GitHub: [@SAKTHIM-collab](https://github.com/SAKTHIM-collab)  

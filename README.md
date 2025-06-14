DOMAIN SPECIFIC TASK :

# Disc Battle Game

A strategic board game with power-ups and timer-based gameplay.

## Demo-Video
https://drive.google.com/file/d/1EePQPd1-Ej-gjS5YsLG434batghGJSTF/view?usp=drivesdk

## Features
- Two-player turn-based gameplay
- Power-ups system (Extra Time, Reduce Time, Extra Turn)
- Timer-based turns (30 seconds per turn)
- Column blocking mechanism
- undo functionality
- Dark/Light theme toggle
- Sound effects
- Score tracking and leaderboard

## How to Run

1. Clone the repository or download the files
2. Open `index.html` in a modern web browser
3. The game will start automatically

## Game Controls

- Click on a column to place your disc
- Use power-ups by clicking their respective buttons
- Toggle dark/light theme using the theme switch
- Enable/disable sound using the sound toggle
- Undo moves using the undo button
- Replay moves using the replay button

## Power-ups

Each player starts with:
- 2 Extra Time power-ups (adds 10 seconds)
- 2 Reduce Time power-ups (reduces opponent's time by 5 seconds)
- 1 Extra Turn power-up

## Game Rules

1. Players take turns placing discs in columns
2. The first player to connect 4 of their discs wins
3. Each turn is limited to 30 seconds
4. Players can block columns using their power-ups
5. The game ends in a draw if all columns are filled without a winner

## Technical Requirements

- Modern web browser (Chrome, Firefox, Safari)
- JavaScript enabled
- HTML5 Canvas support
- Audio support for sound effects

## Technologies Used
- HTML5
- CSS3
- JavaScript
- Particles.js for background effects


## Credits
- Sound effects from Mixkit
- Font Awesome icons
- Particles.js library

## License
This project is for educational purposes and is not licensed for commercial use.

## Contributing
Feel free to fork this repository and submit pull requests for improvements.

## Support
For questions or issues, please open an issue in the repository.

THANK YOU!




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



Author

Sakthi M
GitHub: [@SAKTHIM-collab](https://github.com/SAKTHIM-collab)  

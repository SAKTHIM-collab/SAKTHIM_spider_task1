
document.addEventListener('DOMContentLoaded', function() {
    const needle = document.querySelector('.needle');
    const stopButton = document.getElementById('stop-button');
    const scoreDisplay = document.getElementById('score-value');
    const twoPlayerBtn = document.getElementById('two-player-btn');
    const playerTurnDisplay = document.getElementById('player-turn');
    const winnerPopup = document.getElementById('winner-popup');
    const winnerText = document.getElementById('winner-text');
    const closePopup = document.getElementById('close-popup');
    
    let angle = 0;
    let direction = 1;
    let speed = 1;
    let animationId;
    let isRunning = true;
    let isTwoPlayerMode = false;
    let currentPlayer = 1;
    let player1Score = 0;
    let player2Score = 0;
    
    startGame();
    
    stopButton.addEventListener('click', stopNeedle);
    twoPlayerBtn.addEventListener('click', toggleTwoPlayerMode);
    closePopup.addEventListener('click', closeWinnerPopup);
    
    document.addEventListener('keydown', function(e) {
        if (e.code === 'Space') {
            e.preventDefault();
            stopNeedle();
        }
    });
    
    function startGame() {
        isRunning = true;
        animateNeedle();
    }
    
    function animateNeedle() {
        const speedFactor = 0.5 + Math.abs(Math.sin(angle * Math.PI / 180));
        speed = speedFactor * 2;
        
        angle += speed * direction;
        
        if (angle >= 180 || angle <= 0) {
            direction *= -1;
        }
        
        needle.style.transform = `translateX(-50%) rotate(${angle}deg)`;
        animationId = requestAnimationFrame(animateNeedle);
    }
    
    function stopNeedle() {
        if (!isRunning) return;
        
        cancelAnimationFrame(animationId);
        isRunning = false;
        
        const deviation = Math.abs(90 - angle);
        let score = Math.max(0, Math.round(100 - (deviation * (100/90))));
        
        animateScore(0, score);
        
        if (isTwoPlayerMode) {
            if (currentPlayer === 1) {
                player1Score = score;
                currentPlayer = 2;
                playerTurnDisplay.textContent = "Player 2's Turn";
                playerTurnDisplay.style.color = "#00FF00";
            } else {
                player2Score = score;
                showWinner();
                return;
            }
            
            setTimeout(resetNeedle, 2000);
        } else {
            setTimeout(resetNeedle, 2000);
        }
    }
    
    function animateScore(start, end) {
        let current = start;
        const increment = end > start ? 1 : -1;
        const timer = setInterval(() => {
            current += increment;
            scoreDisplay.textContent = current;
            
            if (current >= 80) {
                scoreDisplay.style.color = "#00FF00";
            } else if (current >= 50) {
                scoreDisplay.style.color = "#FFA500";
            } else {
                scoreDisplay.style.color = "#FF0000";
            }
            
            if (current === end) {
                clearInterval(timer);
                if (end >= 90) {
                    celebrate();
                }
            }
        }, 20);
    }
    
    function resetNeedle() {
        angle = 0;
        direction = 1;
        isRunning = true;
        animateNeedle();
    }
    
    function toggleTwoPlayerMode() {
        isTwoPlayerMode = !isTwoPlayerMode;
        
        if (isTwoPlayerMode) {
            twoPlayerBtn.textContent = "SINGLE PLAYER MODE";
            twoPlayerBtn.style.background = "linear-gradient(to bottom, #FF4500, #FF6347)";
            playerTurnDisplay.textContent = "Player 1's Turn";
            playerTurnDisplay.style.color = "#FF0000";
            currentPlayer = 1;
            player1Score = 0;
            player2Score = 0;
            scoreDisplay.textContent = "0";
        } else {
            twoPlayerBtn.textContent = "2-PLAYER MODE";
            twoPlayerBtn.style.background = "linear-gradient(to bottom, #4169E1, #1E90FF)";
            playerTurnDisplay.textContent = "";
        }
        
        resetNeedle();
    }
    
    function showWinner() {
        let winnerMessage = "";
        
        if (player1Score > player2Score) {
            winnerMessage = "Player 1 Wins!";
        } else if (player2Score > player1Score) {
            winnerMessage = "Player 2 Wins!";
        } else {
            winnerMessage = "It's a Tie!";
        }
        
        winnerText.textContent = winnerMessage + ` (${player1Score}-${player2Score})`;
        winnerPopup.style.display = "flex";
        celebrate();
    }
    
    function closeWinnerPopup() {
        winnerPopup.style.display = "none";
        currentPlayer = 1;
        playerTurnDisplay.textContent = "Player 1's Turn";
        playerTurnDisplay.style.color = "#FF0000";
        resetNeedle();
    }
    
    function celebrate() {
        const colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF'];
        const gameContainer = document.querySelector('.game-container');
        
        for (let i = 0; i < 30; i++) {
            const confetti = document.createElement('div');
            confetti.className = 'confetti';
            confetti.style.left = Math.random() * 100 + '%';
            confetti.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
            confetti.style.transform = `rotate(${Math.random() * 360}deg)`;
            gameContainer.appendChild(confetti);
            
            setTimeout(() => {
                confetti.style.top = '-20px';
                confetti.style.opacity = '0';
                confetti.style.transform = `rotate(${Math.random() * 360}deg) translateY(${Math.random() * 100 + 100}px)`;
            }, 10);
            
            setTimeout(() => {
                confetti.remove();
            }, 2000);
        }
        
        if (typeof Audio !== 'undefined') {
            const audio = new Audio('https://assets.mixkit.co/sfx/preview/mixkit-winning-chimes-2015.mp3');
            audio.play().catch(e => console.log("Audio play failed:", e));
        }
    }
    
    const style = document.createElement('style');
    style.textContent = `
        .confetti {
            position: absolute;
            width: 10px;
            height: 10px;
            background-color: #FF0000;
            top: 0;
            opacity: 1;
            transition: all 1s ease-out;
            z-index: 100;
        }
    `;
    document.head.appendChild(style);
});


document.addEventListener('DOMContentLoaded', () => {
    const ROWS = 6;
    const COLS = 7;
    const PLAYER1 = 'red';
    const PLAYER2 = 'yellow';
    const TURN_TIME = 30; 
    
    let board = [];
    let currentPlayer = PLAYER1;
    let gameOver = false;
    let blockedColumn = null;
    let player1Time = TURN_TIME;
    let player2Time = TURN_TIME;
    let timerInterval;
    let movesCount = 0;
    
    const gameBoard = document.getElementById('gameBoard');
    const player1Timer = document.getElementById('player1Timer');
    const player2Timer = document.getElementById('player2Timer');
    const player1TimerBar = document.getElementById('player1TimerBar');
    const player2TimerBar = document.getElementById('player2TimerBar');
    const statusMessage = document.getElementById('statusMessage');
    const blockMessage = document.getElementById('blockMessage');
    const blockColumns = document.getElementById('blockColumns');
    const gameOverModal = document.getElementById('gameOverModal');
    const modalTitle = document.getElementById('modalTitle');
    const modalMessage = document.getElementById('modalMessage');
    const winnerDisc = document.getElementById('winnerDisc');
    const scoreTable = document.querySelector('#scoreTable tbody');
    const playAgainBtn = document.getElementById('playAgainBtn');
    const newGameBtn = document.getElementById('newGameBtn');
    const themeToggle = document.getElementById('themeToggle');
    const soundToggle = document.getElementById('soundToggle');
    const movesCountElement = document.getElementById('movesCount');
    
    const dropSound = document.getElementById('dropSound');
    const winSound = document.getElementById('winSound');
    const blockSound = document.getElementById('blockSound');
    const countdownSound = document.getElementById('countdownSound');
    
    particlesJS('particles-js', {
        particles: {
            number: { value: 80, density: { enable: true, value_area: 800 } },
            color: { value: "#6c5ce7" },
            shape: { type: "circle" },
            opacity: { value: 0.5, random: true },
            size: { value: 3, random: true },
            line_linked: { enable: true, distance: 150, color: "#6c5ce7", opacity: 0.3, width: 1 },
            move: { enable: true, speed: 2, direction: "none", random: true, straight: false, out_mode: "out" }
        },
        interactivity: {
            detect_on: "canvas",
            events: {
                onhover: { enable: true, mode: "repulse" },
                onclick: { enable: true, mode: "push" }
            }
        }
    });
    
    initGame();
    
    function initGame() {
        board = Array(ROWS).fill().map(() => Array(COLS).fill(null));
        
        gameBoard.innerHTML = '';
        
        for (let row = 0; row < ROWS; row++) {
            for (let col = 0; col < COLS; col++) {
                const cell = document.createElement('div');
                cell.classList.add('cell');
                cell.dataset.row = row;
                cell.dataset.col = col;
                cell.addEventListener('click', () => handleCellClick(row, col));
                gameBoard.appendChild(cell);
            }
        }
        
        currentPlayer = PLAYER1;
        gameOver = false;
        blockedColumn = null;
        player1Time = TURN_TIME;
        player2Time = TURN_TIME;
        movesCount = 0;
        
        updatePlayerInfo();
        updateMovesCount();
        statusMessage.textContent = "Player 1's turn";
        startTimer();
        
        document.querySelectorAll('.indicator').forEach(indicator => {
            indicator.addEventListener('click', () => {
                const col = parseInt(indicator.dataset.col);
                const emptyRow = findEmptyRow(col);
                if (emptyRow !== -1 && !(blockedColumn !== null && col === blockedColumn)) {
                    handleCellClick(emptyRow, col);
                }
            });
        });
    }
    
    function handleCellClick(row, col) {
        if (gameOver) return;
        
        const column = col;
        const emptyRow = findEmptyRow(column);
        
        if (emptyRow === -1 || (blockedColumn !== null && column === blockedColumn)) {
            return; 
        }
        
        board[emptyRow][column] = currentPlayer;
        const cell = document.querySelector(`.cell[data-row="${emptyRow}"][data-col="${column}"]`);
        cell.classList.add(currentPlayer);
        
        playSound(dropSound);
        
        movesCount++;
        updateMovesCount();
        
        if (checkWin(emptyRow, column)) {
            endGame(`${currentPlayer === PLAYER1 ? 'Player 1' : 'Player 2'} wins!`);
            return;
        }
        
        if (checkDraw()) {
            endGame("It's a draw!");
            return;
        }
        
        switchPlayer();
    }
    
    function findEmptyRow(column) {
        for (let row = ROWS - 1; row >= 0; row--) {
            if (board[row][column] === null) {
                return row;
            }
        }
        return -1; 
    }
    
    function checkWin(row, col) {
        const directions = [
            [0, 1],  
            [1, 0],  
            [1, 1],  
            [1, -1]  
        ];
        
        for (const [dx, dy] of directions) {
            let count = 1;
            
            count += countDirection(row, col, dx, dy);
            
            count += countDirection(row, col, -dx, -dy);
            
            if (count >= 4) {
                highlightWinningDiscs(row, col, dx, dy);
                highlightWinningDiscs(row, col, -dx, -dy);
                return true;
            }
        }
        
        return false;
    }
    
    function highlightWinningDiscs(row, col, dx, dy) {
        let r = row + dx;
        let c = col + dy;
        
        while (r >= 0 && r < ROWS && c >= 0 && c < COLS && board[r][c] === currentPlayer) {
            const cell = document.querySelector(`.cell[data-row="${r}"][data-col="${c}"]`);
            cell.classList.add('winning-disc');
            r += dx;
            c += dy;
        }
    }
    
    function countDirection(row, col, dx, dy) {
        let count = 0;
        let r = row + dx;
        let c = col + dy;
        
        while (r >= 0 && r < ROWS && c >= 0 && c < COLS && board[r][c] === currentPlayer) {
            count++;
            r += dx;
            c += dy;
        }
        
        return count;
    }
    
    function checkDraw() {
        return board.every(row => row.every(cell => cell !== null));
    }
    
    function switchPlayer() {
        clearInterval(timerInterval);
        
        currentPlayer = currentPlayer === PLAYER1 ? PLAYER2 : PLAYER1;
        
        if (currentPlayer === PLAYER2) {
            showBlockOptions();
        } else {
            blockedColumn = null;
            updatePlayerInfo();
            statusMessage.textContent = "Player 1's turn";
            startTimer();
        }
    }
    
    function showBlockOptions() {
        const cells = document.querySelectorAll('.cell');
        cells.forEach(cell => cell.style.pointerEvents = 'none');
        
        blockColumns.innerHTML = '';
        
        const availableColumns = [];
        for (let col = 0; col < COLS; col++) {
            if (findEmptyRow(col) !== -1 && col !== blockedColumn) {
                availableColumns.push(col);
            }
        }
        
        if (availableColumns.length <= 1) {
            statusMessage.textContent = "No columns available to block. Player 2's turn";
            blockMessage.style.display = 'none';
            updatePlayerInfo();
            startTimer();
            cells.forEach(cell => cell.style.pointerEvents = 'auto');
            return;
        }
        
        blockMessage.style.display = 'block';
        statusMessage.textContent = "Player 2, select a column to block";
        
        availableColumns.forEach(col => {
            const blockCol = document.createElement('div');
            blockCol.classList.add('block-column');
            blockCol.textContent = col + 1;
            blockCol.addEventListener('click', () => {
                document.querySelectorAll('.block-column').forEach(btn => {
                    btn.classList.remove('selected');
                });
                blockCol.classList.add('selected');
                setTimeout(() => selectBlockColumn(col), 300);
            });
            blockColumns.appendChild(blockCol);
        });
    }
    
    function selectBlockColumn(col) {
        blockedColumn = col;
        
        playSound(blockSound);
        
        blockMessage.style.display = 'none';
        
        const cells = document.querySelectorAll('.cell');
        cells.forEach(cell => cell.style.pointerEvents = 'auto');
        
        updatePlayerInfo();
        statusMessage.textContent = "Player 2's turn";
        startTimer();
    }
    
    function updatePlayerInfo() {
        const player1Info = document.querySelector('.player-info.player1');
        const player2Info = document.querySelector('.player-info.player2');
        
        if (currentPlayer === PLAYER1) {
            player1Info.classList.add('active');
            player2Info.classList.remove('active');
            
            player1Info.querySelector('.disc').classList.add('pulse');
            player2Info.querySelector('.disc').classList.remove('pulse');
        } else {
            player1Info.classList.remove('active');
            player2Info.classList.add('active');
            
            player2Info.querySelector('.disc').classList.add('pulse');
            player1Info.querySelector('.disc').classList.remove('pulse');
        }
        
        player1Timer.textContent = player1Time;
        player2Timer.textContent = player2Time;
        
        updateTimerBars();
    }
    
    function updateTimerBars() {
        player1TimerBar.style.background = `linear-gradient(90deg, var(--primary-color) ${(player1Time / TURN_TIME) * 100}%, rgba(255, 255, 255, 0.2) ${(player1Time / TURN_TIME) * 100}%)`;
        player2TimerBar.style.background = `linear-gradient(90deg, var(--secondary-color) ${(player2Time / TURN_TIME) * 100}%, rgba(255, 255, 255, 0.2) ${(player2Time / TURN_TIME) * 100}%)`;
    }
    
    function updateMovesCount() {
        movesCountElement.textContent = movesCount;
    }
    
    function startTimer() {
        let timeLeft = currentPlayer === PLAYER1 ? player1Time : player2Time;
        const timerElement = currentPlayer === PLAYER1 ? player1Timer : player2Timer;
        const timerBarElement = currentPlayer === PLAYER1 ? player1TimerBar : player2TimerBar;
        
        timerBarElement.style.animation = 'none';
        timerBarElement.offsetHeight; 
        timerBarElement.style.animation = `timerBar ${timeLeft}s linear forwards`;
        
        timerInterval = setInterval(() => {
            timeLeft--;
            
            if (currentPlayer === PLAYER1) {
                player1Time = timeLeft;
            } else {
                player2Time = timeLeft;
            }
            
            timerElement.textContent = timeLeft;
            
            if (timeLeft <= 5 && timeLeft > 0) {
                playSound(countdownSound);
            }
            
            updateTimerBars();
            
            if (timeLeft <= 0) {
                clearInterval(timerInterval);
                endGame(`${currentPlayer === PLAYER1 ? 'Player 2' : 'Player 1'} wins by timeout!`);
            }
        }, 1000);
    }
    
    function endGame(message) {
        gameOver = true;
        clearInterval(timerInterval);
        
        if (!message.includes("draw")) {
            playSound(winSound);
            triggerConfetti();
            
            if (message.includes("Player 1")) {
                winnerDisc.className = 'winner-disc red';
            } else {
                winnerDisc.className = 'winner-disc yellow';
            }
        } else {
            winnerDisc.style.display = 'none';
        }
        
        modalTitle.textContent = "Game Over";
        modalMessage.textContent = message;
        
        updateLeaderboard(message);
        
        gameOverModal.style.display = 'flex';
    }
    
    function triggerConfetti() {
        const confettiSettings = {
            particleCount: 150,
            spread: 70,
            origin: { y: 0.6 },
            colors: ['#6c5ce7', '#00cec9', '#fd79a8', '#fdcb6e', '#ff7675']
        };
        
        if (currentPlayer === PLAYER1) {
            confettiSettings.origin.x = 0.25;
        } else {
            confettiSettings.origin.x = 0.75;
        }
        
        confetti(confettiSettings);
    }
    
    function updateLeaderboard(message) {
        let scores = JSON.parse(localStorage.getItem('discBattleScores')) || {
            player1: { wins: 0, games: 0 },
            player2: { wins: 0, games: 0 }
        };
        
        scores.player1.games++;
        scores.player2.games++;
        
        if (message.includes("Player 1 wins")) {
            scores.player1.wins++;
        } else if (message.includes("Player 2 wins")) {
            scores.player2.wins++;
        }
        
        localStorage.setItem('discBattleScores', JSON.stringify(scores));
        
        const player1WinRate = scores.player1.games > 0 ? (scores.player1.wins / scores.player1.games * 100).toFixed(1) : 0;
        const player2WinRate = scores.player2.games > 0 ? (scores.player2.wins / scores.player2.games * 100).toFixed(1) : 0;
        
        scoreTable.innerHTML = `
            <tr>
                <td>Player 1</td>
                <td>${scores.player1.wins}</td>
                <td>${player1WinRate}%</td>
            </tr>
            <tr>
                <td>Player 2</td>
                <td>${scores.player2.wins}</td>
                <td>${player2WinRate}%</td>
            </tr>
        `;
    }
    
    function playSound(sound) {
        if (soundToggle.checked) {
            sound.currentTime = 0;
            sound.play();
        }
    }
    
    playAgainBtn.addEventListener('click', () => {
        gameOverModal.style.display = 'none';
        initGame();
    });
    
    newGameBtn.addEventListener('click', () => {
        localStorage.setItem('discBattleScores', JSON.stringify({
            player1: { wins: 0, games: 0 },
            player2: { wins: 0, games: 0 }
        }));
        
        gameOverModal.style.display = 'none';
        initGame();
    });
    
    themeToggle.addEventListener('change', () => {
        document.body.classList.toggle('dark-theme');
        localStorage.setItem('discBattleTheme', themeToggle.checked ? 'dark' : 'light');
    });
    
    if (localStorage.getItem('discBattleTheme') === 'dark') {
        document.body.classList.add('dark-theme');
        themeToggle.checked = true;
    }
    
    if (!localStorage.getItem('discBattleScores')) {
        localStorage.setItem('discBattleScores', JSON.stringify({
            player1: { wins: 0, games: 0 },
            player2: { wins: 0, games: 0 }
        }));
    }
});

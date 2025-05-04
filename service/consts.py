html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>music generation</title>
    <style>
        html, body {
            height: 100%;
            margin: 0;
        }
        body {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center; 
            font-family: Arial, sans-serif;
            text-align: center;
        }
        button {
            margin: 5px;
            padding: 10px 20px;
            font-size: 16px;
        }
        audio {
            display: block;
            margin: 20px auto;
            width: 80%;
        }
        #spinner {
            border: 8px solid #f3f3f3;
            border-top: 8px solid #3498db;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
            display: none;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        input[type=number] {
            width: 100px;
            padding: 5px;
            font-size: 16px;
            margin: 10px;
        }
    </style>
</head>
<body>
    <h1>MusicGen Web Service</h1>
    <div>
        <label for="maxTokens">Max tokens:</label>
        <input id="maxTokens" type="number" value="500" min="1" max="2000">
    </div>
    <div>
        <button onclick="generate('classical')">Classical</button>
        <button onclick="generate('jazz')">Jazz</button>
        <button onclick="generate('edm')">EDM</button>
        <button onclick="generate('pop')">Pop</button>
    </div>
    <div id="spinner"></div>
    <audio id="player" controls></audio>
    <script>
        async function generate(genre) {
            const buttons = document.querySelectorAll('button');
            const spinner = document.getElementById('spinner');
            const player = document.getElementById('player');
            const maxTokens = document.getElementById('maxTokens').value;
            buttons.forEach(b => b.disabled = true);
            spinner.style.display = 'block';
            try {
                const response = await fetch(`/generate/${genre}?max_tokens=${encodeURIComponent(maxTokens)}`);
                if (!response.ok) throw new Error(`Error: ${response.statusText}`);
                spinner.style.display = 'none';
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                player.src = url;
                await player.play();
            } catch (e) {
                spinner.style.display = 'none';
                alert(e);
            } finally {
                buttons.forEach(b => b.disabled = false);
            }
        }
    </script>
</body>
</html>
"""

task = "text-to-audio"
GENRE_TOKENS = {
    'pop': '<GENRE_POP>',
    'jazz': '<GENRE_JAZZ>',
    'edm': '<GENRE_EDM>',
    'classical': '<GENRE_CLASSICAL>'
}

SPECIAL_TOKENS = ["PAD", "BOS", "EOS", "MASK"] + list(GENRE_TOKENS.values())
task = "text-to-audio"
actual_checkpoint = "ckpt-epoch10"
infer_params = {"do_sample": True, "temperature": 1.1, "max_new_tokens": 500}
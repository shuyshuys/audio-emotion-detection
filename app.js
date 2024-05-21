async function uploadAudio() {
    const input = document.getElementById('audioInput');
    if (input.files.length === 0) {
        alert('Please select an audio file.');
        return;
    }

    const file = input.files[0];
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    document.getElementById('result').innerText = `Prediction: ${result.prediction}`;
}

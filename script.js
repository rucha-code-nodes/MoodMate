const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const outputText = document.getElementById('outputText');
const movieList = document.getElementById('movieList');

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
  })
  .catch(err => {
    console.error("Camera access error:", err);
  });

function captureAndSend() {
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataURL = canvas.toDataURL('image/jpeg');

  fetch('/detect_emotion', {
    method: 'POST',
    body: JSON.stringify({ image: dataURL }),
    headers: {
      'Content-Type': 'application/json'
    }
  })
  .then(res => res.json())
  .then(data => {
    outputText.innerText = `Emotion: ${data.emotion}`;
    if (data.movies.length > 0) {
      movieList.innerText = `Movie Suggestions: ${data.movies.join(', ')}`;
    } else {
      movieList.innerText = `No suggestions available`;
    }
  });
}



function updateResults(emotion, movies) {
  const outputText = document.getElementById("outputText");
  const movieList = document.getElementById("movieList");
  const movieBox = document.getElementById("movieBox");

  outputText.textContent = `😊 Emotion: ${emotion}`;
  
  if (movies && movies.length > 0) {
    movieList.textContent = `🎬 Movie Suggestions: ${movies.join(", ")}`;
    movieBox.classList.remove("hidden");
    movieBox.classList.add("visible");
  } else {
    movieBox.classList.remove("visible");
    movieBox.classList.add("hidden");
  }
}


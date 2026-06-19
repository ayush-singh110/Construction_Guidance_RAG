const askBtn = document.getElementById('askBtn');
const clearBtn = document.getElementById('clearBtn');
const questionEl = document.getElementById('question');
const answerCard = document.getElementById('answerCard');
const answerText = document.getElementById('answerText');
const loading = document.getElementById('loading');

askBtn.addEventListener('click', async () => {
  const q = questionEl.value.trim();
  if (!q) return;
  loading.classList.remove('d-none');
  answerCard.classList.add('d-none');
  answerText.textContent = '';

  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q })
    });

    const data = await res.json();
    if (data.error) {
      answerText.textContent = 'Error: ' + data.error;
    } else {
      answerText.textContent = data.answer || 'No answer returned.';
    }
    answerCard.classList.remove('d-none');
  } catch (e) {
    answerText.textContent = 'Request failed: ' + e.message;
    answerCard.classList.remove('d-none');
  } finally {
    loading.classList.add('d-none');
  }
});

clearBtn.addEventListener('click', () => {
  questionEl.value = '';
  answerCard.classList.add('d-none');
  answerText.textContent = '';
});

// Frontend logic for retrieval-only RAG search

const searchForm = document.getElementById('searchForm');
const queryInput = document.getElementById('query');
const topKInput = document.getElementById('topK');
const thresholdInput = document.getElementById('threshold');
const searchBtn = document.getElementById('searchBtn');
const loadingDiv = document.getElementById('loading');
const errorDiv = document.getElementById('error');
const resultsDiv = document.getElementById('results');

searchForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const query = queryInput.value.trim();
    if (!query) {
        showError('Please enter a search query');
        return;
    }

    const topK = parseInt(topKInput.value) || 5;
    const threshold = parseFloat(thresholdInput.value) || 0.3;

    // Show loading state
    setLoading(true);
    hideError();
    clearResults();

    try {
        const response = await fetch('/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                top_k: topK,
                similarity_threshold: threshold
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Search failed');
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        showError(error.message || 'An error occurred during search');
    } finally {
        setLoading(false);
    }
});

function setLoading(isLoading) {
    if (isLoading) {
        loadingDiv.classList.remove('hidden');
        searchBtn.disabled = true;
    } else {
        loadingDiv.classList.add('hidden');
        searchBtn.disabled = false;
    }
}

function showError(message) {
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');
}

function hideError() {
    errorDiv.classList.add('hidden');
}

function clearResults() {
    resultsDiv.innerHTML = '';
}

function displayResults(data) {
    clearResults();

    if (!data.results || data.results.length === 0) {
        resultsDiv.innerHTML = `
            <div class="no-results">
                <div class="no-results-title">No results found</div>
                <div class="no-results-text">Try adjusting the similarity threshold or using different search terms.</div>
            </div>
        `;
        return;
    }

    // Add results header
    const resultsHeader = document.createElement('div');
    resultsHeader.className = 'results-header';
    const resultsTitle = document.createElement('div');
    resultsTitle.className = 'results-title';
    resultsTitle.textContent = `Found ${data.results.length} result${data.results.length === 1 ? '' : 's'}`;
    resultsHeader.appendChild(resultsTitle);
    resultsDiv.appendChild(resultsHeader);

    data.results.forEach((result, index) => {
        const card = document.createElement('div');
        card.className = 'result-card';

        const header = document.createElement('div');
        header.className = 'result-header';

        const idSpan = document.createElement('span');
        idSpan.className = 'result-id';
        idSpan.textContent = `ID: ${result.id}`;

        const similaritySpan = document.createElement('span');
        similaritySpan.className = 'similarity-badge';
        similaritySpan.textContent = `Similarity: ${result.similarity.toFixed(4)}`;

        header.appendChild(idSpan);
        header.appendChild(similaritySpan);

        const textDiv = document.createElement('div');
        textDiv.className = 'result-text';
        // Show first 500 characters, with ellipsis if longer
        const text = result.text.length > 500 
            ? result.text.substring(0, 500) + '...' 
            : result.text;
        textDiv.textContent = text;

        card.appendChild(header);
        card.appendChild(textDiv);

        resultsDiv.appendChild(card);
    });
}


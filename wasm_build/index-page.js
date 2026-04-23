import { loadOpenBlas } from './openblas-loader.js';

const tests = {
    daxpy: { cardClass: 'level1', badge: 'Level 1 · D', title: 'DAXPY', expected: 35.0, tol: 1e-9, description: '<strong>y = α·x + y</strong><br>α=2, x=[1..5], y=[1..1]' },
    ddot:  { cardClass: 'level1', badge: 'Level 1 · D', title: 'DDOT',  expected: 70.0, tol: 1e-9, description: '<strong>result = x·y</strong><br>x=[1,2,3,4], y=[5,6,7,8]' },
    dscal: { cardClass: 'level1', badge: 'Level 1 · D', title: 'DSCAL', expected: 45.0, tol: 1e-9, description: '<strong>x = α·x</strong><br>α=3, x=[1,2,3,4,5]' },
    dnrm2: { cardClass: 'level1', badge: 'Level 1 · D', title: 'DNRM2', expected: 7.0,  tol: 1e-9, description: '<strong>‖x‖₂</strong><br>x=[2,6,3]' },
    dcopy: { cardClass: 'level1', badge: 'Level 1 · D', title: 'DCOPY', expected: 6.0,  tol: 1e-9, description: '<strong>y = x</strong><br>stride=2, soma esperada=6.0' },
    dswap: { cardClass: 'level1', badge: 'Level 1 · D', title: 'DSWAP', expected: 15.0, tol: 1e-9, description: '<strong>swap(x, y)</strong><br>stride=2, soma de x=15.0' },

    saxpy: { cardClass: 'level1', badge: 'Level 1 · S', title: 'SAXPY', expected: 15.0, tol: 1e-5, description: '<strong>y = α·x + y</strong><br>stride=2, α=2, soma=15.0' },
    sdot:  { cardClass: 'level1', badge: 'Level 1 · S', title: 'SDOT',  expected: 32.0, tol: 1e-5, description: '<strong>result = x·y</strong><br>stride=2, esperado=32.0' },
    snrm2: { cardClass: 'level1', badge: 'Level 1 · S', title: 'SNRM2', expected: 5.0,  tol: 1e-5, description: '<strong>‖x‖₂</strong><br>x usado=[3,4], stride=2' },
    sscal: { cardClass: 'level1', badge: 'Level 1 · S', title: 'SSCAL', expected: 3.0,  tol: 1e-5, description: '<strong>x = α·x</strong><br>stride=2, α=0.5, soma=3.0' },
    scopy: { cardClass: 'level1', badge: 'Level 1 · S', title: 'SCOPY', expected: 6.0,  tol: 1e-5, description: '<strong>y = x</strong><br>stride=2, soma=6.0' },
    sswap: { cardClass: 'level1', badge: 'Level 1 · S', title: 'SSWAP', expected: 15.0, tol: 1e-5, description: '<strong>swap(x, y)</strong><br>stride=2, soma de x=15.0' },

    caxpy: { cardClass: 'complex1', badge: 'Level 1 · C', title: 'CAXPY', expected: 4.0,  tol: 1e-5, description: '<strong>y = α·x + y</strong><br>complexo, stride=2, soma Re+Im=4.0' },
    cdotc: { cardClass: 'complex1', badge: 'Level 1 · C', title: 'CDOTC', expected: 70.0, tol: 1e-5, description: '<strong>result = conj(x)·y</strong><br>retorna parte real' },
    cdotu: { cardClass: 'complex1', badge: 'Level 1 · C', title: 'CDOTU', expected: 68.0, tol: 1e-5, description: '<strong>result = x·y</strong><br>retorna parte imaginária' },
    cscal: { cardClass: 'complex1', badge: 'Level 1 · C', title: 'CSCAL', expected: 14.0, tol: 1e-5, description: '<strong>x = α·x</strong><br>complexo, soma Re+Im=14.0' },
    ccopy: { cardClass: 'complex1', badge: 'Level 1 · C', title: 'CCOPY', expected: 4.0,  tol: 1e-5, description: '<strong>y = x</strong><br>complexo, stride=2, soma=4.0' },
    cswap: { cardClass: 'complex1', badge: 'Level 1 · C', title: 'CSWAP', expected: 9.0,  tol: 1e-5, description: '<strong>swap(x, y)</strong><br>complexo, stride=2, soma de x=9.0' },

    dgemv: { cardClass: 'level2', badge: 'Level 2 · D', title: 'DGEMV', expected: 21.0, tol: 1e-9, description: '<strong>y = α·A·x + β·y</strong><br>A=3×2, x=[1,1]' },
    dger:  { cardClass: 'level2', badge: 'Level 2 · D', title: 'DGER',  expected: 18.0, tol: 1e-9, description: '<strong>A = α·x·yᵀ + A</strong><br>x=[1,2], y=[1,2,3]' },

    dgemm: { cardClass: 'level3', badge: 'Level 3 · D', title: 'DGEMM', expected: 134.0, tol: 1e-9, description: '<strong>C = α·A·B + β·C</strong><br>matrizes 2×2' },
    dtrsm: { cardClass: 'level3', badge: 'Level 3 · D', title: 'DTRSM', expected: 5.0,   tol: 1e-9, description: '<strong>A·X = α·B</strong><br>triangular lower 2×2' },
    dsymm: { cardClass: 'level3', badge: 'Level 3 · D', title: 'DSYMM', expected: 11.0,  tol: 1e-9, description: '<strong>C = α·A·B + β·C</strong><br>A simétrica, B=I' },

    zaxpy: { cardClass: 'complex1', badge: 'Level 1 · Z', title: 'ZAXPY', expected: 7.0,  tol: 1e-9, description: '<strong>y = α·x + y</strong><br>α=1+1i, soma Re+Im=7.0' },
    zdotc: { cardClass: 'complex1', badge: 'Level 1 · Z', title: 'ZDOTC', expected: 70.0, tol: 1e-9, description: '<strong>result = conj(x)·y</strong><br>retorna parte real' },
    zdotu: { cardClass: 'complex1', badge: 'Level 1 · Z', title: 'ZDOTU', expected: 68.0, tol: 1e-9, description: '<strong>result = x·y</strong><br>retorna parte imaginária' },
    zscal: { cardClass: 'complex1', badge: 'Level 1 · Z', title: 'ZSCAL', expected: 16.0, tol: 1e-9, description: '<strong>x = α·x</strong><br>complexo, soma Re+Im=16.0' },
    zcopy: { cardClass: 'complex1', badge: 'Level 1 · Z', title: 'ZCOPY', expected: 4.0,  tol: 1e-9, description: '<strong>y = x</strong><br>complexo, stride=2, soma=4.0' },
    zswap: { cardClass: 'complex1', badge: 'Level 1 · Z', title: 'ZSWAP', expected: 9.0,  tol: 1e-9, description: '<strong>swap(x, y)</strong><br>complexo, stride=2, soma de x=9.0' },
    zgemv: { cardClass: 'complex2', badge: 'Level 2 · Z', title: 'ZGEMV', expected: 2.0,  tol: 1e-9, description: '<strong>y = α·A·x + β·y</strong><br>A=2×2 unitária' },
    zgemm: { cardClass: 'complex3', badge: 'Level 3 · Z', title: 'ZGEMM', expected: 36.0, tol: 1e-9, description: '<strong>C = α·A·B + β·C</strong><br>A=2×2, B=I complexa' },
};

let openBlas;

function resultId(testName) {
    return `result-${testName}`;
}

function buttonId(testName) {
    return `btn-${testName}`;
}

function renderTestCards() {
    const grid = document.getElementById('tests-grid');
    grid.innerHTML = Object.entries(tests).map(([name, test]) => `
        <div class="test-card ${test.cardClass}">
            <span class="level-badge">${test.badge}</span>
            <h3>${test.title}</h3>
            <p>${test.description}<br>Expected: ${test.expected}</p>
            <div class="result pending" id="${resultId(name)}">Pending...</div>
            <button onclick="runTest('${name}')" id="${buttonId(name)}" disabled>Run Test</button>
        </div>
    `).join('');
}

function logConsole(msg, type = 'info') {
    const consoleEl = document.getElementById('console');
    const p = document.createElement('p');
    p.className = type;
    p.textContent = '> ' + msg;
    consoleEl.appendChild(p);
    consoleEl.scrollTop = consoleEl.scrollHeight;
}

function updateStatus(msg, type = 'loading') {
    const status = document.getElementById('status');
    status.className = 'status-box ' + type;
    status.innerHTML = '<p>' + msg + '</p>';
    logConsole(msg, type);
}

function enableTestButtons() {
    Object.keys(tests).forEach((testName) => {
        document.getElementById(buttonId(testName)).disabled = false;
    });
    document.getElementById('runBtn').disabled = false;
}

function clearResults() {
    Object.keys(tests).forEach((testName) => {
        const resultDiv = document.getElementById(resultId(testName));
        resultDiv.className = 'result pending';
        resultDiv.textContent = 'Pending...';
    });
    document.getElementById('console').innerHTML = '';
    logConsole('Results cleared', 'info');
}

function ensureReady() {
    if (!openBlas) {
        throw new Error('WebAssembly module is not initialized yet.');
    }
    return openBlas;
}

function runTest(testName) {
    try {
        const api = ensureReady();
        const { expected, tol = 1e-6 } = tests[testName];
        const resultDiv = document.getElementById(resultId(testName));
        resultDiv.className = 'result pending';
        resultDiv.textContent = 'Running...';

        logConsole(`Executing ${testName.toUpperCase()}...`, 'info');

        const result = api.callTest(testName);
        const ok = Math.abs(result - expected) <= tol;
        resultDiv.className = 'result ' + (ok ? 'success' : 'error');
        resultDiv.textContent = `Result: ${result.toFixed(4)} (Expected: ${expected}, Tol: ${tol})`;
        logConsole(`${testName.toUpperCase()}: ${result.toFixed(4)} [${ok ? 'PASS' : 'FAIL'}]`, ok ? 'info' : 'error');
    } catch (err) {
        const resultDiv = document.getElementById(resultId(testName));
        resultDiv.className = 'result error';
        resultDiv.textContent = 'Error: ' + err.message;
        logConsole(`Error in ${testName.toUpperCase()}: ${err.message}`, 'error');
    }
}

async function runAllTests() {
    logConsole('Starting full test suite...', 'info');
    for (const testName of Object.keys(tests)) {
        runTest(testName);
        await new Promise((resolve) => setTimeout(resolve, 100));
    }
    logConsole('Test suite completed!', 'info');
}

async function initializeIndexPage() {
    renderTestCards();
    updateStatus('📥 Loading WebAssembly module...', 'loading');
    try {
        openBlas = await loadOpenBlas({
            print: (text) => logConsole(String(text), 'info'),
            printErr: (text) => logConsole(String(text), 'error'),
            onAbort: (message) => updateStatus('❌ Failed to load WebAssembly: ' + message, 'error')
        });
        updateStatus('✅ WebAssembly module loaded successfully!', 'success');
        enableTestButtons();
        logConsole('WASM exports available: ' + Object.keys(tests).map((k) => '_test_' + k).join(', '));
    } catch (error) {
        updateStatus('❌ Failed to initialize WebAssembly module: ' + error.message, 'error');
    }
}

window.runTest = runTest;
window.runAllTests = runAllTests;
window.clearResults = clearResults;

initializeIndexPage();

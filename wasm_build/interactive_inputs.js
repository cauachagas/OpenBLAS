import { loadOpenBlas } from './openblas-loader.js';

const presets = {
    daxpy: [
        {
            n: 5,
            alpha: 2,
            incx: 1,
            incy: 1,
            x: '1, 2, 3, 4, 5',
            y: '1, 1, 1, 1, 1'
        },
        {
            n: 3,
            alpha: 1.5,
            incx: 2,
            incy: 2,
            x: '2, 99, -1, 99, 4',
            y: '10, 77, 20, 77, 30'
        }
    ],
    ddot: [
        {
            n: 4,
            incx: 1,
            incy: 1,
            x: '1, 2, 3, 4',
            y: '5, 6, 7, 8'
        },
        {
            n: 3,
            incx: 2,
            incy: 2,
            x: '1, 99, 2, 99, 3',
            y: '4, 77, 5, 77, 6'
        }
    ],
    dnrm2: [
        {
            n: 3,
            incx: 1,
            x: '2, 6, 3'
        },
        {
            n: 2,
            incx: 2,
            x: '3, 99, 4'
        }
    ]
};

let openBlas;

function logConsole(message, type = 'info') {
    const consoleEl = document.getElementById('console');
    const line = document.createElement('p');
    line.className = type;
    line.textContent = '> ' + message;
    consoleEl.appendChild(line);
    consoleEl.scrollTop = consoleEl.scrollHeight;
}

function setStatus(message, type = 'loading') {
    const box = document.getElementById('status');
    box.className = 'status-box ' + type;
    box.textContent = message;
    logConsole(message, type === 'error' ? 'error' : 'info');
}

function parseVector(text) {
    const items = text
        .split(/[\s,;]+/)
        .map((item) => item.trim())
        .filter(Boolean)
        .map((item) => Number(item));

    if (items.some((value) => Number.isNaN(value))) {
        throw new Error('Um dos valores do vetor nao e numerico.');
    }

    return items;
}

function requiredLength(n, inc) {
    return ((n - 1) * inc) + 1;
}

function ensureVectorLength(values, n, inc, label) {
    const minLength = requiredLength(n, inc);
    if (values.length < minLength) {
        throw new Error(`${label} precisa ter pelo menos ${minLength} valores para n=${n} e inc=${inc}.`);
    }
}

function formatVector(values) {
    return '[' + values.map((value) => Number(value).toFixed(6)).join(', ') + ']';
}

function readNumber(id) {
    const value = Number(document.getElementById(id).value);
    if (Number.isNaN(value)) {
        throw new Error(`Campo ${id} nao e numerico.`);
    }
    return value;
}

function ensureReady() {
    if (!openBlas) {
        throw new Error('Modulo WebAssembly ainda nao foi inicializado.');
    }
    return openBlas;
}

function runDaxpy() {
    try {
        const api = ensureReady();
        const n = readNumber('daxpy-n');
        const alpha = readNumber('daxpy-alpha');
        const incx = readNumber('daxpy-incx');
        const incy = readNumber('daxpy-incy');
        const x = parseVector(document.getElementById('daxpy-x').value);
        const y = parseVector(document.getElementById('daxpy-y').value);

        ensureVectorLength(x, n, incx, 'x');
        ensureVectorLength(y, n, incy, 'y');

        const result = api.daxpy({ n, alpha, x, incx, y, incy });

        document.getElementById('result-daxpy').textContent =
            'Entrada x: ' + formatVector(x) + '\n' +
            'Entrada y original: ' + formatVector(y) + '\n' +
            'Saida y: ' + formatVector(result);
        logConsole('DAXPY executado com sucesso.', 'info');
    } catch (error) {
        document.getElementById('result-daxpy').textContent = 'Erro: ' + error.message;
        logConsole('Erro no DAXPY: ' + error.message, 'error');
    }
}

function runDdot() {
    try {
        const api = ensureReady();
        const n = readNumber('ddot-n');
        const incx = readNumber('ddot-incx');
        const incy = readNumber('ddot-incy');
        const x = parseVector(document.getElementById('ddot-x').value);
        const y = parseVector(document.getElementById('ddot-y').value);

        ensureVectorLength(x, n, incx, 'x');
        ensureVectorLength(y, n, incy, 'y');

        const result = api.ddot({ n, x, incx, y, incy });

        document.getElementById('result-ddot').textContent =
            'Entrada x: ' + formatVector(x) + '\n' +
            'Entrada y: ' + formatVector(y) + '\n' +
            'Resultado ddot: ' + result.toFixed(12);
        logConsole('DDOT executado com sucesso.', 'info');
    } catch (error) {
        document.getElementById('result-ddot').textContent = 'Erro: ' + error.message;
        logConsole('Erro no DDOT: ' + error.message, 'error');
    }
}

function runDnrm2() {
    try {
        const api = ensureReady();
        const n = readNumber('dnrm2-n');
        const incx = readNumber('dnrm2-incx');
        const x = parseVector(document.getElementById('dnrm2-x').value);

        ensureVectorLength(x, n, incx, 'x');

        const result = api.dnrm2({ n, x, incx });

        document.getElementById('result-dnrm2').textContent =
            'Entrada x: ' + formatVector(x) + '\n' +
            'Resultado ||x||_2: ' + result.toFixed(12);
        logConsole('DNRM2 executado com sucesso.', 'info');
    } catch (error) {
        document.getElementById('result-dnrm2').textContent = 'Erro: ' + error.message;
        logConsole('Erro no DNRM2: ' + error.message, 'error');
    }
}

function loadPreset(name, index) {
    const preset = presets[name][index];
    Object.entries(preset).forEach(([key, value]) => {
        const element = document.getElementById(`${name}-${key}`);
        if (element) {
            element.value = value;
        }
    });
    logConsole(`Preset ${index + 1} aplicado em ${name.toUpperCase()}.`, 'warn');
}

function enableButtons() {
    document.getElementById('run-daxpy').disabled = false;
    document.getElementById('run-ddot').disabled = false;
    document.getElementById('run-dnrm2').disabled = false;
}

async function initializeInteractivePage() {
    setStatus('Loading parameterized BLAS wrappers...', 'loading');
    try {
        openBlas = await loadOpenBlas({
            print: (text) => logConsole(String(text), 'info'),
            printErr: (text) => logConsole(String(text), 'error'),
            onAbort: (message) => setStatus('Failed to load WebAssembly: ' + message, 'error')
        });
        setStatus('WebAssembly module ready to receive inputs.', 'success');
        enableButtons();
        logConsole('Modular loader initialized for GitHub Pages.', 'info');
    } catch (error) {
        setStatus('Failed to initialize WebAssembly module: ' + error.message, 'error');
    }
}

window.runDaxpy = runDaxpy;
window.runDdot = runDdot;
window.runDnrm2 = runDnrm2;
window.loadPreset = loadPreset;

initializeInteractivePage();

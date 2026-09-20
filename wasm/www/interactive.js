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
        throw new Error('Vector contains non-numeric values.');
    }
    return items;
}

function requiredLength(n, inc) {
    return ((n - 1) * inc) + 1;
}

function ensureVectorLength(values, n, inc, label) {
    const minLength = requiredLength(n, inc);
    if (values.length < minLength) {
        throw new Error(`${label} requires at least ${minLength} values for n=${n} and inc=${inc}.`);
    }
}

function formatVector(values) {
    return '[' + values.map((val) => Number(val).toFixed(6)).join(', ') + ']';
}

function readNumber(id) {
    const value = Number(document.getElementById(id).value);
    if (Number.isNaN(value)) {
        throw new Error(`Field ${id} is not a valid number.`);
    }
    return value;
}

function runDaxpy() {
    try {
        if (!openBlas) throw new Error('WebAssembly module not ready.');
        const n = readNumber('daxpy-n');
        const alpha = readNumber('daxpy-alpha');
        const incx = readNumber('daxpy-incx');
        const incy = readNumber('daxpy-incy');
        const x = parseVector(document.getElementById('daxpy-x').value);
        const y = parseVector(document.getElementById('daxpy-y').value);

        ensureVectorLength(x, n, incx, 'Vector x');
        ensureVectorLength(y, n, incy, 'Vector y');

        const t0 = performance.now();
        const result = openBlas.daxpy({ n, alpha, x, incx, y, incy });
        const elapsed = (performance.now() - t0).toFixed(3);

        document.getElementById('result-daxpy').textContent =
            `✓ Executed in ${elapsed} ms\nInput x: ${formatVector(x)}\nOriginal y: ${formatVector(y)}\nOutput y: ${formatVector(result)}`;
        logConsole(`DAXPY completed in ${elapsed} ms`, 'info');
    } catch (error) {
        document.getElementById('result-daxpy').textContent = 'Error: ' + error.message;
        logConsole('DAXPY Error: ' + error.message, 'error');
    }
}

function runDdot() {
    try {
        if (!openBlas) throw new Error('WebAssembly module not ready.');
        const n = readNumber('ddot-n');
        const incx = readNumber('ddot-incx');
        const incy = readNumber('ddot-incy');
        const x = parseVector(document.getElementById('ddot-x').value);
        const y = parseVector(document.getElementById('ddot-y').value);

        ensureVectorLength(x, n, incx, 'Vector x');
        ensureVectorLength(y, n, incy, 'Vector y');

        const t0 = performance.now();
        const result = openBlas.ddot({ n, x, incx, y, incy });
        const elapsed = (performance.now() - t0).toFixed(3);

        document.getElementById('result-ddot').textContent =
            `✓ Executed in ${elapsed} ms\nResult (dot product): ${result.toFixed(12)}`;
        logConsole(`DDOT completed: ${result} in ${elapsed} ms`, 'info');
    } catch (error) {
        document.getElementById('result-ddot').textContent = 'Error: ' + error.message;
        logConsole('DDOT Error: ' + error.message, 'error');
    }
}

function runDnrm2() {
    try {
        if (!openBlas) throw new Error('WebAssembly module not ready.');
        const n = readNumber('dnrm2-n');
        const incx = readNumber('dnrm2-incx');
        const x = parseVector(document.getElementById('dnrm2-x').value);

        ensureVectorLength(x, n, incx, 'Vector x');

        const t0 = performance.now();
        const result = openBlas.dnrm2({ n, x, incx });
        const elapsed = (performance.now() - t0).toFixed(3);

        document.getElementById('result-dnrm2').textContent =
            `✓ Executed in ${elapsed} ms\nResult (Euclidean norm ‖x‖₂): ${result.toFixed(12)}`;
        logConsole(`DNRM2 completed: ${result} in ${elapsed} ms`, 'info');
    } catch (error) {
        document.getElementById('result-dnrm2').textContent = 'Error: ' + error.message;
        logConsole('DNRM2 Error: ' + error.message, 'error');
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
    logConsole(`Applied preset ${index + 1} for ${name.toUpperCase()}`, 'warn');
}

function enableButtons() {
    document.getElementById('run-daxpy').disabled = false;
    document.getElementById('run-ddot').disabled = false;
    document.getElementById('run-dnrm2').disabled = false;
}

async function initializeInteractivePage() {
    setStatus('Loading OpenBLAS WebAssembly module...', 'loading');
    try {
        openBlas = await loadOpenBlas({
            print: (text) => logConsole(String(text), 'info'),
            printErr: (text) => logConsole(String(text), 'error'),
            onAbort: (message) => setStatus('WebAssembly Aborted: ' + message, 'error')
        });
        setStatus('✓ OpenBLAS WebAssembly module loaded and ready.', 'success');
        enableButtons();
    } catch (error) {
        setStatus('Failed to load WebAssembly module: ' + error.message, 'error');
    }
}

window.runDaxpy = runDaxpy;
window.runDdot = runDdot;
window.runDnrm2 = runDnrm2;
window.loadPreset = loadPreset;

initializeInteractivePage();

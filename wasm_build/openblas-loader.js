import createOpenBlasModule from './test.mjs';

let openBlasPromise;

function resolveLocateFile(path) {
    return new URL(path, import.meta.url).href;
}

function createApi(module) {
    function writeFloat64Array(ptr, values) {
        for (let index = 0; index < values.length; index += 1) {
            const ok = module._wasm_set_f64(ptr, index, values[index]);
            if (!ok) {
                throw new Error(`Failed to write value at index ${index} into the WebAssembly buffer.`);
            }
        }
    }

    function readFloat64Array(ptr, length) {
        const values = new Array(length);
        for (let index = 0; index < length; index += 1) {
            values[index] = module._wasm_get_f64(ptr, index);
        }
        return values;
    }

    function allocFloat64(length, initialValues) {
        const ptr = module._malloc(length * Float64Array.BYTES_PER_ELEMENT);
        if (!ptr) {
            throw new Error('Failed to allocate memory in the WebAssembly heap.');
        }

        if (initialValues) {
            writeFloat64Array(ptr, initialValues);
        } else {
            writeFloat64Array(ptr, new Array(length).fill(0));
        }

        return {
            ptr,
            length,
            free() {
                module._free(ptr);
            }
        };
    }

    function callTest(testName) {
        const fn = module[`_test_${testName}`];
        if (typeof fn !== 'function') {
            throw new Error(`Function _test_${testName} not found`);
        }
        return fn();
    }

    function daxpy({ n, alpha, x, incx = 1, y, incy = 1 }) {
        const xAlloc = allocFloat64(x.length, x);
        const yAlloc = allocFloat64(y.length, y);
        try {
            const ok = module._wasm_daxpy(n, alpha, xAlloc.ptr, incx, yAlloc.ptr, incy);
            if (!ok) {
                throw new Error('wasm_daxpy returned an error status.');
            }
            return readFloat64Array(yAlloc.ptr, y.length);
        } finally {
            yAlloc.free();
            xAlloc.free();
        }
    }

    function ddot({ n, x, incx = 1, y, incy = 1 }) {
        const xAlloc = allocFloat64(x.length, x);
        const yAlloc = allocFloat64(y.length, y);
        const resultAlloc = allocFloat64(1);
        try {
            const ok = module._wasm_ddot(n, xAlloc.ptr, incx, yAlloc.ptr, incy, resultAlloc.ptr);
            if (!ok) {
                throw new Error('wasm_ddot returned an error status.');
            }
            return readFloat64Array(resultAlloc.ptr, 1)[0];
        } finally {
            resultAlloc.free();
            yAlloc.free();
            xAlloc.free();
        }
    }

    function dnrm2({ n, x, incx = 1 }) {
        const xAlloc = allocFloat64(x.length, x);
        const resultAlloc = allocFloat64(1);
        try {
            const ok = module._wasm_dnrm2(n, xAlloc.ptr, incx, resultAlloc.ptr);
            if (!ok) {
                throw new Error('wasm_dnrm2 returned an error status.');
            }
            return readFloat64Array(resultAlloc.ptr, 1)[0];
        } finally {
            resultAlloc.free();
            xAlloc.free();
        }
    }

    return {
        module,
        callTest,
        daxpy,
        ddot,
        dnrm2,
        hasExport(name) {
            return typeof module[name] === 'function';
        }
    };
}

export async function loadOpenBlas(options = {}) {
    if (!openBlasPromise) {
        openBlasPromise = createOpenBlasModule({
            locateFile: resolveLocateFile,
            print: options.print,
            printErr: options.printErr,
            onAbort: options.onAbort
        }).then((module) => createApi(module));
    }

    return openBlasPromise;
}

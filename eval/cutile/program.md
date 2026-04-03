# cuTile Kernel Implementation

## Goal
Implement a cuTile kernel that passes the test in `test/`.

## Setup
1. Read the cuTile skill in `.opencode/skill/cutile-python/SKILL.md` to understand the cuTile API
2. Read `problem-spec.yaml` to understand what kernel to implement
3. Read the test file in `test/` to understand expected inputs, outputs, and function signatures
4. Initialize git tracking: `git init && git add -A && git commit -m "initial workspace"`

## Experiment Loop
Repeat until the test passes or you have made 15 attempts:

1. **Write or modify** `solution.py` with your cuTile kernel implementation
2. **Commit your attempt**: `git add solution.py && git commit -m "attempt N: <brief description of change>"`
3. **Run the test**:
   ```
   cd /testbed && PYTHONPATH=/testbed python -m pytest test/ -x -v 2>&1 | tee -a results.log
   ```
4. **Check the result**:
   - If all tests **PASSED**: append `FINAL: PASS attempts=N` to results.log, then stop
   - If tests **FAILED**: read the error message carefully, understand what went wrong, then go back to step 1

## Rules
- Only create or modify `solution.py` — never modify files in `test/`
- Use the cuTile API: `import cuda.tile as ct`
- Every solution must include `@ct.kernel` decorator and `ct.launch()` call
- Use `ct.float32` accumulators for numerical stability in reductions and matmul
- Always commit before running the test so we can track every iteration
- Read error messages carefully — they tell you exactly what went wrong
- If you get a compilation error, check your ct.* API usage against the skill doc
- If you get a numerical mismatch, check accumulator dtype and reduction order

## When Done
Append your final status to results.log:
```
FINAL: PASS attempts=<N> kernel=<kernel_name>
```
or if you could not solve it after 15 attempts:
```
FINAL: FAIL attempts=15 kernel=<kernel_name>
```

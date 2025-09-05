module DirectHiGHS

using HiGHS: Highs_create, Highs_destroy, Highs_setBoolOptionValue, Highs_addCol,
    Highs_changeColIntegrality, Highs_changeObjectiveSense, Highs_getObjectiveSense,
    Highs_addRow, Highs_run, Highs_getSolution, kHighsObjSenseMinimize, kHighsVarTypeInteger,
    Highs_passLp, kHighsMatrixFormatColwise, Highs_getModelStatus, Highs_changeColsCostBySet,
    Highs_changeCoeff, Highs_changeColsBoundsBySet
using SparseArrays
import HiGHS
import MathOptInterface as MOI

"""
    solve_lp()

Solves a small linear programming problem using the HiGHS C API.

minimize 1.0x + 1.0y
subject to:
  5.0 <= 1.0x + 2.0y <= 15.0
  6.0 <= 3.0x + 2.0y
  0.0 <= x <= 4.0
  1.0 <= y
  y is an integer
"""
function solve_lp()
    highs = Highs_create()
    ret = Highs_setBoolOptionValue(highs, "log_to_console", false)
    @assert ret == 0

    Highs_addCol(highs, 1.0, 0.0, 4.0, 0, C_NULL, C_NULL)
    Highs_addCol(highs, 1.0, 1.0, Inf, 0, C_NULL, C_NULL)
    Highs_changeColIntegrality(highs, 1, kHighsVarTypeInteger)
    Highs_changeObjectiveSense(highs, kHighsObjSenseMinimize)

    senseP = Ref{Cint}(0)
    Highs_getObjectiveSense(highs, senseP)
    @assert senseP[] == kHighsObjSenseMinimize

    Highs_addRow(highs, 5.0, 15.0, 2, Cint[0, 1], [1.0, 2.0])
    Highs_addRow(highs, 6.0, Inf, 2, Cint[0, 1], [3.0, 2.0])
    Highs_run(highs)

    col_value = zeros(Cdouble, 2)
    Highs_getSolution(highs, col_value, C_NULL, C_NULL, C_NULL)

    println("x = ", col_value[1])
    println("y = ", col_value[2])

    Highs_destroy(highs)
end

"""
    solve_sparse_lp()

Solves a small sparse linear programming problem using the HiGHS C API.

The problem is defined as:
```math
\\min \\sum_{i=1}^6 c_i x_i
```
subject to:
```math
A x \\le b
```
```math
x \\ge 0
```
where:
```math
c = [-1, -2, -3, -4, -5, -6]
```
```math
A = \\begin{pmatrix}
1 & 1 & 0 & 0 & 0 & 0 \\\\
0 & 1 & 1 & 0 & 0 & 0 \\\\
0 & 0 & 1 & 1 & 0 & 0 \\\\
0 & 0 & 0 & 1 & 1 & 0 \\\\
0 & 0 & 0 & 0 & 1 & 1
\\end{pmatrix}
```
```math
b = [2, 2, 2, 2, 2]
```
"""
function solve_sparse_lp()
    highs = Highs_create()
    ret = Highs_setBoolOptionValue(highs, "log_to_console", false)
    @assert ret == 0

    num_col = 6
    num_row = 5

    col_cost = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
    col_lower = zeros(num_col)
    col_upper = fill(Inf, num_col)

    row_lower = -Inf * ones(num_row)
    row_upper = [2.0, 2.0, 2.0, 2.0, 2.0]

    A = sparse([1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
               [1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
               [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    sense = kHighsObjSenseMinimize
    offset = 0.0

    col_value = zeros(Cdouble, num_col)
    col_dual = zeros(Cdouble, num_col)
    row_value = zeros(Cdouble, num_row)
    row_dual = zeros(Cdouble, num_row)
    col_basis_status = zeros(Cint, num_col)
    row_basis_status = zeros(Cint, num_row)
    model_status = Ref{Cint}()

    A_colptr = Cint.(A.colptr .- 1)
    A_rowval = Cint.(A.rowval .- 1)

    ret = Highs_passLp(highs, num_col, num_row, length(A.nzval), kHighsMatrixFormatColwise, sense, offset,
                 col_cost, col_lower, col_upper, row_lower, row_upper,
                 A_colptr, A_rowval, A.nzval)
    @assert ret == 0

    ret = Highs_run(highs)
    @assert ret == 0

    ret = Highs_getSolution(highs, col_value, col_dual, row_value, row_dual)
    @assert ret == 0

    println("\nSparse LP solution:")
    for i in 1:num_col
        println("x[$i] = ", col_value[i])
    end

    # Modify the problem
    new_costs = col_cost .+ rand(num_col) .* 2.0
    ret = Highs_changeColsCostBySet(highs, 6, Cint[0, 1, 2, 3, 4, 5], new_costs)
    @assert ret == 0

    new_lower = col_lower .+ rand(num_col) .* 0.5
    new_upper = col_upper
    ret = Highs_changeColsBoundsBySet(highs, 6, Cint[0, 1, 2, 3, 4, 5], new_lower, new_upper)
    @assert ret == 0

    ret = Highs_run(highs)
    @assert ret == 0

    ret = Highs_getSolution(highs, col_value, col_dual, row_value, row_dual)
    @assert ret == 0

    println("\nModified sparse LP solution:")
    for i in 1:num_col
        println("x[$i] = ", col_value[i])
    end

    Highs_destroy(highs)
end

println("Solving non-sparse LP:")
# solve_lp()
solve_sparse_lp()

function solve_moi_lp()
    optimizer = HiGHS.Optimizer()
    MOI.set(optimizer, MOI.Silent(), true)

    num_col = 6
    num_row = 5

    col_cost = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
    col_lower = zeros(num_col)
    col_upper = fill(Inf, num_col)

    row_lower = -Inf * ones(num_row)
    row_upper = [2.0, 2.0, 2.0, 2.0, 2.0]

    A = sparse([1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
               [1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
               [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    x = MOI.add_variables(optimizer, num_col)
    for i in 1:num_col
        MOI.add_constraint(optimizer, x[i], MOI.GreaterThan(col_lower[i]))
        if col_upper[i] < Inf
            MOI.add_constraint(optimizer, x[i], MOI.LessThan(col_upper[i]))
        end
    end

    for i in 1:num_row
        terms = MOI.ScalarAffineTerm{Float64}[]
        for j in 1:num_col
            if A[i, j] != 0
                push!(terms, MOI.ScalarAffineTerm(A[i, j], x[j]))
            end
        end
        if row_lower[i] > -Inf
            MOI.add_constraint(optimizer, MOI.ScalarAffineFunction(terms, 0.0), MOI.GreaterThan(row_lower[i]))
        end
        if row_upper[i] < Inf
            MOI.add_constraint(optimizer, MOI.ScalarAffineFunction(terms, 0.0), MOI.LessThan(row_upper[i]))
        end
    end

    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(col_cost, x), 0.0))
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(optimizer)

    col_value = MOI.get(optimizer, MOI.VariablePrimal(), x)

    println("\nMOI LP solution:")
    for i in 1:num_col
        println("x[$i] = ", col_value[i])
    end
end

solve_moi_lp()

end # module

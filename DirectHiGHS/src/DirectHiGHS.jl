module DirectHiGHS

using HiGHS: Highs_create, Highs_destroy, Highs_setBoolOptionValue, Highs_addCol,
    Highs_changeColIntegrality, Highs_changeObjectiveSense, Highs_getObjectiveSense,
    Highs_addRow, Highs_run, Highs_getSolution, kHighsObjSenseMinimize, kHighsVarTypeInteger

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

solve_lp()

end # module

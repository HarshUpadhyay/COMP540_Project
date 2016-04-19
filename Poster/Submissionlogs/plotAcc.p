set terminal  pdf enhanced color
set output "LogAcc.pdf"
set xlabel "# of epochs"
set ylabel "Accuracy (ratio)"
set xrange [0:300]
set yrange [0.2:1]
set datafile separator "&"
set style line 1 lc rgb '#0060ad' lt 1 lw 1 pt 7 ps 0.05 
set style line 2 lc rgb '#dd181f' lt 1 lw 1 pt 7 ps 0.05 
set title "Accuracy on Training and Validation datasets"
plot "Log1Data.txt" using 1:4 title "Training data (Model:1)",  "Log1Data.txt" using 1:5 title "Validation data  (Model:1)" 
plot "Log4Data.txt" using 1:4 title "Training data (Model:2)",  "Log4Data.txt" using 1:5 title "Validation data (Model:2)" 
plot "Log7Data.txt" using 1:4 title "Training data (Model:3)",  "Log7Data.txt" using 1:5 title "Validation data (Model:3)" 
plot "Log8Data.txt" using 1:4 title "Training data (Kaught22)",  "Log8Data.txt" using 1:5 title "Validation data (4th K-Submission)" 
plot "OldMachineLog1Data.txt" using 1:4 title "Training data (Model:1 Ubuntu)",  "OldMachineLog1Data.txt" using 1:5 title "Validation data (Model:1 Ubuntu)" 
plot "Log2Data.txt" using 1:4 title "Training data (FC(256, 0.5))",  "Log2Data.txt" using 1:5 title "Validation data (FC(256, 0.5))"
plot "Log3Data.txt" using 1:4 title "Training data (FC(512, 0.5))",  "Log3Data.txt" using 1:5 title "Validation data (FC(512, 0.5))"

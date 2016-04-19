set terminal  pdf enhanced color
set output "LogLoss.pdf"
set xlabel "# of epochs"
set ylabel "Loss"
set xrange [0:300]
set yrange [0:2]
set datafile separator "&"
set style line 1 lc rgb '#0060ad' lt 1 lw 1 pt 7 ps 0.05 
set style line 2 lc rgb '#dd181f' lt 1 lw 1 pt 7 ps 0.05 
set title "Accuracy on Training and Validation datasets"
plot "Log1Data.txt" using 1:2 title "Training data (Model:1)",  "Log1Data.txt" using 1:3 title "Validation data  (Model:1)" 
plot "Log4Data.txt" using 1:2 title "Training data (Model:2)",  "Log4Data.txt" using 1:3 title "Validation data (Model:2)" 
plot "Log7Data.txt" using 1:2 title "Training data (Model:3)",  "Log7Data.txt" using 1:3 title "Validation data (Model:3)"
plot "Log8Data.txt" using 1:2 title "Training data (Kaught22)",  "Log8Data.txt" using 1:3 title "Validation data (Kaught22)" 
plot "OldMachineLog1Data.txt" using 1:2 title "Training data (Model:1 Ubuntu)",  "OldMachineLog1Data.txt" using 1:3 title "Validation data (Model:1 Ubuntu)" 

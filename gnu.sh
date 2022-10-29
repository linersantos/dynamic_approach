#!/bin/bash
#este script pega 500 arquivos com os valores de Kn gerados a partir dos eventos da JETSCAPE em classes de centrlidade 0-5% em diante e faz uma extrapolação para estimar o valor de Kn em regime ultracentral (E.g. 0-1%).
rm 'fit.log'
rm 'erro.dat'
rm 'saida.dat'
rm 'a1.dat'
rm 'e1.dat'

rm 'a2.dat'
rm 'e2.dat'

rm 'a3.dat'
rm 'e3.dat'

rm 'a4.dat'
rm 'e4.dat'

rm 'a5.dat'
rm 'e5.dat'
for i in {0..499..1} #cada arquivo contem os valores de Kn para várias classes de centralidade a partir de 0-5%
	#do cat '/home/liner/doutorado/bayes/liner/Outputs_all_events/Pb-Pb-2760_eccn_vn2_kn_vs_centrality_design_point_'$i'_idf_3.dat' | sed -n '14,19p' >> temp.dat
	do if [ -s /home/liner/doutorado/js-sims-bayes/cms/n0/$i.dat ] #se o arquivo não é vazio
	then
		cat '/home/liner/doutorado/js-sims-bayes/cms/n0/'$i'.dat' | sed -n '2,19p' >> temp.dat #copia cada arquivo da pasta n0 num arquivo temporario temp
	#fi

gnuplot -e "set grid; unset key; set xlabel 'cent. bin' font'times,15'; set ylabel 'K_n' font'times,15';
set fit errorvariables;
f2(x) = a2 + b2*x + c2*x**2;
f3(x) = a3 + b3*x + c3*x**2;
f4(x) = a4 + b4*x + c4*x**2;
f5(x) = a5 + b5*x + c5*x**2;

g2(x) = d2 + e2*x + f2*x**2;
g3(x) = d3 + e3*x + f3*x**2;
g4(x) = d4 + e4*x + f4*x**2;
g5(x) = d5 + e5*x + f5*x**2;

fit f2(x) 'temp.dat' u 1:2 via a2,b2,c2;
fit f3(x) 'temp.dat' u 1:4 via a3,b3,c3;
fit f4(x) 'temp.dat' u 1:6 via a4,b4,c4;
fit f5(x) 'temp.dat' u 1:8 via a5,b5,c5;

fit g2(x) 'temp.dat' u 1:3 via d2,e2,f2;
fit g3(x) 'temp.dat' u 1:6 via d3,e3,f3;
fit g4(x) 'temp.dat' u 1:9 via d4,e4,f4;
fit g5(x) 'temp.dat' u 1:12 via d5,e5,f5;

set print 'a1.dat';
print 1,a2 + b2*1 + c2*1, a3 + b3*1 + c3*1, a4 + b4*1 + c4*1, a5 + b5*1 + c5*1, d2 + e2*1 + f2*1, d3 + e3*1 + f3*1, d4 + e4*1 + f4*1**2, d5 + e5*1 + f5*1;
set print 'err.dat';
print 1,a2_err + b2_err + c2_err, a3_err + b3_err + c3_err, a4_err + b4_err + c4_err, a5_err + b5_err + c5_err, d2_err + e2_err + f2_err, d3_err + e3_err + f3_err, d4_err + e4_err + f4_err, d5_err + e5_err + f5_err;
set print 'a2.dat';
print 1, a2 + b2*1 + c2*1, d2 + e2*1 + f2*1, 0, a3 + b3*1 + c3*1, d3 + e3*1 + f3*1, 0 , a4 + b4*1 + c4*1, d4 + e4*1 + f4*1, 0, a5 + b5*1 + c5*1, d5 + e5*1 + f5*1, 0;
exit"



cat 'a1.dat' >> 'saida.dat'
cat 'err.dat' >> 'erro.dat'
cat 'a2.dat' >> '/home/liner/doutorado/js-sims-bayes/cms/n0/'$i'.dat'
#cat 'a3.dat' >> 'e3.dat'
#cat 'a4.dat' >> 'e4.dat'
#cat 'a5.dat' >> 'e5.dat'
fi
rm temp.dat
done
rm a1.dat



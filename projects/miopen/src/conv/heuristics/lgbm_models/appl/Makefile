lgbm_appl.so: main.o quantize.o tu0.o tu1.o tu2.o tu3.o tu4.o tu5.o tu6.o tu7.o
	gcc -shared -O3 -o lgbm_appl.so main.o quantize.o tu0.o tu1.o tu2.o tu3.o tu4.o tu5.o tu6.o tu7.o -std=c99 
main.o: main.c
	gcc -c -O3 -o main.o main.c -fPIC -std=c99 
quantize.o: quantize.c
	gcc -c -O3 -o quantize.o quantize.c -fPIC -std=c99 
tu0.o: tu0.c
	gcc -c -O3 -o tu0.o tu0.c -fPIC -std=c99 
tu1.o: tu1.c
	gcc -c -O3 -o tu1.o tu1.c -fPIC -std=c99 
tu2.o: tu2.c
	gcc -c -O3 -o tu2.o tu2.c -fPIC -std=c99 
tu3.o: tu3.c
	gcc -c -O3 -o tu3.o tu3.c -fPIC -std=c99 
tu4.o: tu4.c
	gcc -c -O3 -o tu4.o tu4.c -fPIC -std=c99 
tu5.o: tu5.c
	gcc -c -O3 -o tu5.o tu5.c -fPIC -std=c99 
tu6.o: tu6.c
	gcc -c -O3 -o tu6.o tu6.c -fPIC -std=c99 
tu7.o: tu7.c
	gcc -c -O3 -o tu7.o tu7.c -fPIC -std=c99 

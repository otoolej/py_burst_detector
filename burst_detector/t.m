

sens_vector = [76.985 62.405 53.297 48.045 39.21 30.3 26.3 20.1 11.4];

spec_vector = [91.285 98.153 99.301 99.798 99.909 99.93 99.95 99.98 99.99];



sens_vector = sens_vector ./ 100;

spec_vector = spec_vector ./ 100;



PAD_ZEROS = 0;

if(PAD_ZEROS)

    sens_vector = fliplr([1 sens_vector 0]);

    spec_vector = fliplr([0 spec_vector 1]);

else

    sens_vector = fliplr(sens_vector);

    spec_vector = fliplr(spec_vector);

end



dauc = trapz(sens_vector, spec_vector);

fprintf('AUC=%g\n', dauc);





figure(1); clf;

plot(sens_vector, spec_vector, '-o');

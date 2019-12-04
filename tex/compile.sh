file_name=ann

pdflatex -synctex=1 -interaction=nonstopmode -output-directory="output" "${file_name}.tex"

# cp output/${file_name}.pdf ${file_name}.pdf
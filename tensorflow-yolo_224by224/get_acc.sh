FILE=bus.txt
while read LINE; do
         python demo_all.py --ifile $LINE 
         echo "This is a line: $LINE"
done < $FILE

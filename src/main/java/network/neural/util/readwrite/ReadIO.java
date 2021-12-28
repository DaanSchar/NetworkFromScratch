package network.neural.util.readwrite;

import network.neural.util.matrix.NDArray;

import java.io.File;
import java.io.FileNotFoundException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class ReadIO {

    private static String COMMA_DELIMITER = ",";

    public static NDArray readCsv(String filename) {
        List<double[]> records = new ArrayList<>();

        try (Scanner scanner = new Scanner(new File("src/main/resources/datasets/" + filename))) {

            while (scanner.hasNextLine())
                records.add(getRecordFromLine(scanner.nextLine()));

        } catch (FileNotFoundException e) {e.printStackTrace();}

        return listToNDArray(records);
    }


    /**
     * splits a line from a csv file into a double array
     */
    private static double[] getRecordFromLine(String line) {
        List<String> values = new ArrayList<>();

        try (Scanner rowScanner = new Scanner(line)) {
            rowScanner.useDelimiter(COMMA_DELIMITER);

            while (rowScanner.hasNext())
                values.add(rowScanner.next());
        }

        return stringToDouble(values);
    }


    /**
     * converts a list of strings to a double array
     */
    private static double[] stringToDouble(List<String> values) {
        double[] valuesDouble = new double[values.size()];

        for (int i = 0; i < values.size(); i++)
            valuesDouble[i] = stringToDouble(values.get(i));

        return valuesDouble;
    }


    /**
     * converts a string to a double
     */
    private static double stringToDouble(String value) {
        double valueDouble;

        value = removeChar(value, '"');

        try {
            valueDouble = Double.parseDouble(value);
        } catch (NumberFormatException e) {
            byte[] stringBytes= value.getBytes(StandardCharsets.UTF_8);
            valueDouble = sum(stringBytes);
        }
        return valueDouble;
    }


    /**
     * returns teh string without containing the character c;
     */
    private static String removeChar(String value, char c) {
        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < value.length(); i++)
            if (value.charAt(i) != c)
                sb.append(value.charAt(i));

        return sb.toString();
    }


    /**
     * returns the sum of a byte array, but excludes the " character
     */
    private static double sum(byte[] string) {
        double sum = 0;

        for (byte b : string)
            if (b != 34) // 34 is the ASCII code for "
                sum += b;

        return sum;
    }


    /**
     * converts the list of double arrays to an NDArray
     */
    private static NDArray listToNDArray(List<double[]> list) {
        double[][] recordsDouble = new double[list.size()][];

        for (int i = 0; i < list.size(); i++)
            recordsDouble[i] = list.get(i);

        return new NDArray(recordsDouble);
    }

}

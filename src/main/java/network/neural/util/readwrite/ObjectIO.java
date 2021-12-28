package network.neural.util.readwrite;

import java.io.*;

public class ObjectIO {

    public static void WriteObjectToFile(Object serObj, String path) {
        try {
            FileOutputStream fileOut = new FileOutputStream(path);
            ObjectOutputStream objectOut = new ObjectOutputStream(fileOut);
            objectOut.writeObject(serObj);
            objectOut.close();
            System.out.println("The Object  was successfully written to a file");

        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public static Object readObjectFromFile(String path) {
        Object obj = null;
        try {
            FileInputStream fileIn = new FileInputStream(path);
            ObjectInputStream objectIn = new ObjectInputStream(fileIn);
            obj = objectIn.readObject();
            objectIn.close();
            System.out.println("The Object  was successfully read from a file");

        } catch (Exception ex) {
            ex.printStackTrace();
        }
        return obj;
    }

}

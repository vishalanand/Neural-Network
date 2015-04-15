import java.io.*;

class NeuralNetwork{
	public static void main(String[] args){
		try{
        	extractFile("NeuralNetwork.py");
        }
        catch(Exception E)
        {}
		try{
			Process p = Runtime.getRuntime().exec("python NeuralNetwork.py -wlimit 0 -dn 4 -word 6 -window 3 -in in.txt -arguments");
			BufferedReader in = new BufferedReader(new InputStreamReader(p.getInputStream()));
			String s;
			while ((s=in.readLine())!=null)
				System.out.println(s);
			Process p_clean = Runtime.getRuntime().exec("rm NeuralNetwork.py");
		}catch(Exception e)
		{
			System.out.println("Error encountered!");
		}
	}
	private static void extractFile(String name) throws IOException{
        ClassLoader cl = NeuralNetwork.class.getClassLoader();
        File target = new File(name);
        if (target.exists())
            return;

        FileOutputStream out = new FileOutputStream(target);
        InputStream in = cl.getResourceAsStream(name);

        byte[] buf = new byte[8*1024];
        int len;
        while((len = in.read(buf)) != -1)
        {
            out.write(buf,0,len);
        }
        out.close();
            in.close();
    }
}

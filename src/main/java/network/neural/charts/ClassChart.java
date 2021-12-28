package network.neural.charts;

import java.awt.Color;
import javax.swing.JFrame;
import javax.swing.WindowConstants;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeriesCollection;

public class ClassChart extends JFrame {

    public ClassChart(String title, XYSeriesCollection classes) {
        super(title);

        // Create dataset

        // Create chart
        JFreeChart chart = ChartFactory.createScatterPlot(
                "Boys VS Girls weight comparison chart",
                "X-Axis", "Y-Axis", classes);
        System.out.println(classes.getX(0,0));


        //Changes background color
        XYPlot plot = (XYPlot)chart.getPlot();
        plot.setBackgroundPaint(new Color(255,228,196));


        // Create Panel
        ChartPanel panel = new ChartPanel(chart);
        setContentPane(panel);
    }



    public void create() {
        this.setSize(800, 400);
        this.setLocationRelativeTo(null);
        this.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
    }

}  
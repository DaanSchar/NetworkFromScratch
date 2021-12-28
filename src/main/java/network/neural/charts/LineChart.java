package network.neural.charts;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.block.BlockBorder;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.title.TextTitle;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.awt.*;

public class LineChart extends JFrame {

    protected String title;
    protected String xLabel;
    protected String yLabel;
    protected double[] x;
    protected double[] y;

    public LineChart(double[] data, String title, String xLabel, String yLabel) {
        this.title = title;
        this.xLabel = xLabel;
        this.yLabel = yLabel;
        this.x = new double[data.length];
        this.y = data;

        for (int i = 0; i < data.length; i++)
            this.x[i] = i;
    }

    public LineChart(double[] x, double[] y, String title, String xLabel, String yLabel) {
        this(x,title,xLabel,yLabel);
        this.x = x;
    }

    public void create() {
        initUI(createDataset(x,y));
    }

    private void initUI(XYDataset dataset) {

        JFreeChart chart = createChart(dataset);

        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setBorder(BorderFactory.createEmptyBorder(15, 15, 15, 15));
        chartPanel.setBackground(Color.white);
        add(chartPanel);

        pack();
        setTitle(title);
        setLocationRelativeTo(null);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    protected XYDataset createDataset(double[] bFacts) {
        XYSeries dataset = new XYSeries("Cost");
        for (int i = 0; i < bFacts.length; i++) {
            dataset.add(i, bFacts[i]);
        }

        return new XYSeriesCollection(dataset);
    }

    protected XYDataset createDataset(double[] x,double[] y) {
        XYSeries dataset = new XYSeries("Cost");
        for (int i = 0; i < x.length; i++) {
            dataset.add(x[i], y[i]);
        }

        return new XYSeriesCollection(dataset);
    }

    protected JFreeChart createChart(XYDataset dataset) {

        JFreeChart chart = ChartFactory.createXYLineChart(
                title,
                xLabel,
                yLabel,
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        XYPlot plot = chart.getXYPlot();

        var renderer = new XYLineAndShapeRenderer();
        renderer.setSeriesPaint(0, Color.RED);
        renderer.setSeriesStroke(0, new BasicStroke(2.0f));

        plot.setRenderer(renderer);
        plot.setBackgroundPaint(Color.white);

        plot.setRangeGridlinesVisible(true);
        plot.setRangeGridlinePaint(Color.BLACK);

        plot.setDomainGridlinesVisible(true);
        plot.setDomainGridlinePaint(Color.BLACK);

        chart.getLegend().setFrame(BlockBorder.NONE);

        chart.setTitle(new TextTitle(title,
                        new Font("Serif", java.awt.Font.BOLD, 18)
                )
        );

        return chart;
    }

}

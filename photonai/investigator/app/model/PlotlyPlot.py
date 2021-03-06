from photonai.investigator.app.model.PlotlyTrace import PlotlyTrace


class PlotlyPlot:
    """ Class which prepares given data to plot in views
    author: Julian Gebker
    version: 1.0.0
    """

    def __init__(self, plot_name: str, title: str, traces: list=None, show_legend: bool=True):
        """ Constructor
        :param plot_name: Name of the div-Element which will show the plot
        :param title: title of the plot
        :param traces: List of plotly traces
        """
        self.plot_name = plot_name
        self.title = title
        if traces is None:
            self.traces = []
        else:
            self.traces = traces
        self.show_legend = show_legend

    def trace_names_to_string(self) -> str:
        """ Returns a comma separated string of containing trace names
        :return: Comma separated string of trace names
        """
        result = ""
        for item in self.traces:
            result += str(item.variable_name) + ","
        return result.rstrip(',')

    def add_trace(self, trace: PlotlyTrace):
        """ Adds given trace to self.traces
        :param trace: Trace to add
        """
        self.traces.append(trace)

    def to_plot(self) -> str:
        """ Returns a string to print script for plot in views
        :return: Javascript String to print in views
        """

        result = ""

        for item in self.traces:
            result += "var " + item.variable_name + " = { x: [" + item.get_x_to_string() + "]"
            result += ", y: [" + item.get_y_to_string() + "]"
            result += ", name: '" + item.variable_name + "'"
            result += ", mode: '" + item.mode + "'"
            result += ", type: '" + item.trace_type + "'"
            result += ", marker: {"
            if item.trace_color:
                result += "color: '" + item.trace_color + "', "
            if item.trace_size != 0:
                result += "size: " + str(item.trace_size)
            result += "}};"

        result += str("var layout = { title: '" + str(self.title) + "', yaxis: {range: [-0.25, 1.25]}")

        if self.show_legend:
            result += ", showlegend: true };"
        else:
            result += ", showlegend: false};"

        result += str("var data = [" + self.trace_names_to_string() + "];")
        result += str("Plotly.newPlot('" + str(self.plot_name) + "', data, layout);")

        return result

    def to_error_plot(self) -> str:
        """ function to print error plot in views
        :return: Javascript String to print in views
        """

        result = ""

        for item in self.traces:
            result += "var " + item.variable_name + " = { x: [" + item.get_x_to_string() + "]"
            result += ", y: [" + item.get_y_to_string() + "]"
            result += ", type: '" + item.trace_type + "'"
            result += ", mode: '" + item.mode + "'"
            result += ", name: '" + item.variable_name + "'"
            result += ", marker: {"
            if item.trace_color:
                result += "color: '" + item.trace_color + "', "
            if item.trace_size != 0:
                result += "size: " + str(item.trace_size)
            result += "}"

            if item.with_error:
                result += ", error_y: {type: 'data', array: [" + item.get_err_y_to_string() + "], visible: true}};"
            else:
                result += "};"

        result += "var layout = {title: '" + self.title + "', width: 600"

        if self.show_legend:
            result += ", showlegend: true };"
        else:
            result += ", showlegend: false};"

        result += str("var data = [" + self.trace_names_to_string() + "];")
        result += "Plotly.newPlot('" + self.plot_name + "', data, layout);"

        return result

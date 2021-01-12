library(shiny)
library(ggplot2)
colnames<-c("xgb", "gbm", "lasso", "null_model","ridge", "decision_tree")
colnames_plot<-c("Employment Type", "Company logo", "Question", "Telecommuting", "Required Experience", "Required Education", "Real Jobs", "Fake Jobs" )
colnames_bal <- c("balanced", "unbalanced")


ui <- navbarPage("Data Science Final Project - Group 5",
                 tabPanel("Raw Data Information",
                          sidebarPanel(
                            h5("Data Variables"),
                            fluidRow (
                              column(5,tableOutput("originTable"))
                            )
                          ),
                          mainPanel(
                            h4("Plot with Factors", align="left"),
                            selectizeInput("Factor_Select",label="Factors",multiple=TRUE,choices=colnames_plot,
                                           options =list(maxItems=1,plugins = list('remove_button', 'drag_drop'))),
                            plotOutput("plotings", width = 800, height= 800)



                          )
                 ),
                 tabPanel("Fraudulent/Unfraudulent",
                        sidebarPanel(selectizeInput("frau_bal",label="Data Set",multiple=TRUE,choices=colnames_bal,
                                                    options =list(maxItems=1,plugins = list('remove_button', 'drag_drop')))
                        ),
                        mainPanel(
                          h4("Balanced V.S Unbalance Data in factor - fraudulent"),
                          plotOutput("fraudulent_plotings", width = 800, height = 800)
                        )


                 ),
                 tabPanel("Performance",
                          sidebarLayout(
                            sidebarPanel(
                              selectizeInput("mySelect",label="Machine Learning Methods",multiple=TRUE,choices=colnames,
                                             options =list(maxItems=1,plugins = list('remove_button', 'drag_drop'))),
                              br(),
                              br(),
                              sliderInput("n", "Number of k-fold",1 ,10, 1),
                            ),

                            mainPanel(
                              h4("ROC with k-fold", align = "left"),
                              plotOutput("plot"),
                              br(),
                              br(),
                              br(),
                              br(),
                              br(),
                              h4("Performance of Machine Learning Method", align = "left"),
                              fluidRow(
                                column(10,tableOutput("table"))
                              )
                            ) # mainPanel
                          )
                 )

)

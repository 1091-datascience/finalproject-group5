library(shiny)
library(ggplot2)

colnames<-c("xgb", "gbm", "lasso", "null_model","ridge", "decision_tree")
colnames_plot<-c("Employment Type", "Company logo", "Question", "Telecommuting", "Required Experience", "Required Education", "Real Jobs", "Fake Jobs" )
ui <- navbarPage("Data Science Final Project - Group 5",
        tabPanel("Raw Data Information",
          sidebarPanel(fluidRow (
            column(5,tableOutput("originTable"))
          )),
          mainPanel(
            h4("Plot with Factors", align="left"),
            selectizeInput("Factor_Select",label="Factors",multiple=TRUE,choices=colnames_plot,
                           options =list(maxItems=1,plugins = list('remove_button', 'drag_drop'))),
            plotOutput("plotings", width = 800, height= 800)



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

server <- function(input, output) {


  output$table <- renderTable({
    if ( input$mySelect == "decision_tree" ) {
      inFile <- normalizePath( file.path('./',paste(input$mySelect, '/cnf_dtree_tv.csv', sep="")))
    }
    else {
      inFile <- normalizePath( file.path('./',paste(input$mySelect, '/cnf_', input$mySelect, '_tv.csv', sep="")))
    }

    if (is.null(inFile))
      return(NULL)
    data <- read.csv(inFile)
    list(src=inFile)
    print(data)

  })

  output$plot <- renderImage({
    if( input$mySelect == "null_model") {
      list(NULL)
    }
    else {
      if ( input$mySelect == "decision_tree" ) {
        filename <- normalizePath(file.path('./',paste(input$mySelect,'/', 'dtree_tv', input$n, '.png', sep='')))
      }
      else {
        filename <- normalizePath(file.path('./',paste(input$mySelect,'/', input$mySelect,'_tv', input$n, '.png', sep='')))
      }

      print(filename)
      list(src=filename)

    }

  }, deleteFile = FALSE)

  output$originTable <- renderTable({
    # df <- read.csv("fake_job_postings.csv", header = T, stringsAsFactors = F, sep=",")
    # df <- df[1:18]
    # counter <- sapply(df, function(x){length(unique(x))})
    # dT <- as.data.frame(counter)
    # colnames(dT) <- c("Varibales")
    # dT1 <- t(dT)
    #
    # print(dT1)
    df <- read.csv("summary.csv", header = T, stringsAsFactors = F, sep=",")
    print(df)




  })

  output$plotings<- renderPlot({

    df <- read.csv("fake_job_postings.csv", header = T, stringsAsFactors = F, sep=",")
    df <- df[1:18]
    #counter <- sapply(df, function(x){length(unique(x))})
    #print(counter)

    df$fraudulent[df$fraudulent=="0"] <- "True Jobs"
    df$fraudulent[df$fraudulent=="1"] <- "Fake Jobs"

    #Employment Type", "Company logo", "Question", "Telecommuting", "Required Experience", "Required Education", "Real Jobs", "Fake Jobs" )
    if (input$Factor_Select == "Employment Type") {

      # employment_type
      df$employment_type[df$employment_type==""] <- NA
      df$employment_type[is.na(df$employment_type)] = "Unknown"
      tbl1 <- with(df, table(employment_type, fraudulent))
      p <- ggplot(as.data.frame(tbl1), aes(x = reorder(employment_type, Freq), y = Freq, fill = fraudulent)) +
        xlab("Employment Types") +
        ylab("Amount") +
        ggtitle("Employment Type Distribution by Target") +
        geom_col(position = 'dodge')+
        geom_text(aes(label=Freq), position=position_dodge(width=0.9), vjust=-0.25)
      print(p)

    }

    else if (input$Factor_Select == "Company logo") {
      # has_company_logo
      df$has_company_logo[df$has_company_logo=="1"] <-"has logo"
      df$has_company_logo[df$has_company_logo=="0"] <-"no logo"

      tbl2 <- with(df, table(has_company_logo, fraudulent))
      p <- ggplot(as.data.frame(tbl2), aes(x = reorder(has_company_logo, Freq), y = Freq, fill = fraudulent)) +
        xlab("Company Logo") +
        ylab("Amount") +
        ggtitle("Has Company logo Distribution by Target") +
        geom_col(position = 'dodge') +
        geom_text(aes(label=Freq), position=position_dodge(width=0.9), vjust=-0.25)
      show(p)
    }

    else if (input$Factor_Select == "Question") {
      # has_questions
      df$has_questions[df$has_questions=="1"] <-"has questions"
      df$has_questions[df$has_questions=="0"] <-"no questions"
      tbl3<- with(df, table(has_questions, fraudulent))
      p <- ggplot(as.data.frame(tbl3), aes(x = reorder(has_questions, Freq), y = Freq, fill = fraudulent)) +
        xlab("Question") +
        ylab("Amount") +
        ggtitle("Has question Distribution by Target") +
        geom_col(position = 'dodge') +
        geom_text(aes(label=Freq), position=position_dodge(width=0.9), vjust=-0.25)
      show(p)
    }

    else if (input$Factor_Select == "Telecommuting") {
      # telecommuting
      df$telecommuting[df$telecommuting=="0"] <- "Without Telecommuting"
      df$telecommuting[df$telecommuting=="1"] <- "With Telecommuting"
      tbl4<- with(df, table(telecommuting, fraudulent))
      p <- ggplot(as.data.frame(tbl4), aes(telecommuting, Freq, fill = fraudulent)) +
        xlab("Telecommuting") +
        ylab("Amount") +
        ggtitle("Telecommuting Distrubution by Target") +
        geom_col(position = 'dodge') +
        geom_text(aes(label=Freq), position=position_dodge(width=0.9), vjust=-0.25)
      show(p)
    }

    else if (input$Factor_Select == "Required Experience") {

      # required experience
      df$required_experience[df$required_experience==""] <- NA
      df$required_experience[is.na(df$required_experience)] = "Unknown"
      tbl5<- with(df, table(required_experience, fraudulent))
      p <- ggplot(as.data.frame(tbl5), aes(x = reorder(required_experience, Freq), y = Freq, fill = fraudulent)) +
        xlab("Required Experience") +
        ylab("Amount") +
        ggtitle("Required Experience Distrubution by Target")+
        geom_col(position = 'dodge') +
        theme(axis.text.x = element_text(angle = 45, size=10, hjust = 1))+
        geom_text(aes(label=Freq), position=position_dodge(width=0.9), vjust=-0.25, size = 3)

      show(p)
    }

    else if (input$Factor_Select == "Required Education") {
      # required_education
      df$required_education[df$required_education==""] <- NA
      df$required_education[is.na(df$required_education)] = "Unknown"
      tbl6<- with(df, table(required_education, fraudulent))
      p <- ggplot(as.data.frame(tbl6), aes(x = reorder(required_education, Freq), y = Freq, fill = fraudulent)) +
        xlab("Degree Types") +
        ylab("Amount") +
        ggtitle("Required Experience Distrubution by Target") +
        geom_col(position = 'dodge') +
        theme(axis.text.x = element_text(angle = 45, size=8, hjust = 1))+
        geom_text(aes(label=Freq), position=position_dodge(width=0.9), vjust=-0.25, size = 3)

      show(p)

    }


    else if (input$Factor_Select == "Real Jobs" ) {
      df$fraudulent[df$fraudulent=="True Jobs"] <- 0
      df$fraudulent[df$fraudulent=="Fake Jobs"] <- 1


      df$industry[df$industry==""] <- NA
      df$industry[is.na(df$industry)] = "Unknown"
      df_real<-as.data.frame(table(df$industry[df$fraudulent==0]))

      df_real_regular_fixed <- df_real[order(df_real$Freq,decreasing=TRUE),][1:10,]
      print(df_real_regular_fixed)
      names(df_real_regular_fixed) <- c("Job_name", "Freq")
      p <- ggplot(df_real_regular_fixed, aes(x = reorder(Job_name, Freq), y= Freq)) +
        ggtitle("Real Jobs") +
        geom_col(position = 'dodge') +
        geom_bar(stat="identity", color='skyblue',fill='steelblue')+
        xlab("Industries") +
        ylab("Amount") +
        theme(axis.text.x = element_text(angle = 45, size=10, hjust = 1))+
        geom_text(aes(label=Freq), position=position_dodge(width=0.9), vjust=-0.25, size = 3)

      show(p)

    }
    else if (input$Factor_Select == "Fake Jobs" ) {
      df$fraudulent[df$fraudulent=="True Jobs"] <- 0
      df$fraudulent[df$fraudulent=="Fake Jobs"] <- 1

      df_fake<-as.data.frame(table(df$industry[df$fraudulent==1]))
      df_fake_regular_fixed <- df_fake[order(df_fake$Freq,decreasing=TRUE),][1:10,]
      print(df_fake_regular_fixed)
      names(df_fake_regular_fixed) <- c("Job_name", "Freq")
      p <- ggplot(df_fake_regular_fixed, aes(x = reorder(Job_name, Freq), y= Freq)) +
        ggtitle("Fake Jobs") +
        geom_col(position = 'dodge') +
        geom_bar(stat="identity", color='skyblue',fill='steelblue')+
        xlab("Industries") +
        ylab("Amount") +
        theme(axis.text.x = element_text(angle = 45, size=10, hjust = 1))+
        geom_text(aes(label=Freq), position=position_dodge(width=0.9), vjust=-0.25, size = 3)

      show(p)
    }
  })

}








# Create Shiny app ----
shinyApp(ui = ui, server = server)

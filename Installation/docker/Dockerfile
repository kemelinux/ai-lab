# azure sql server
# Maintainers: Microsoft Corporation (LuisBosquez and twright-msft on GitHub)
# GitRepo: https://github.com/Microsoft/mssql-docker

# Base OS layer: Latest Ubuntu LTS.
FROM mcr.microsoft.com/azure-sql-edge:latest


# Default SQL Server TCP/Port.
EXPOSE 1433

# Copy all SQL Server runtime files from build drop into image.
#COPY ./install /

# Run SQL Server process.
CMD [ "/opt/mssql/bin/sqlservr" ]
